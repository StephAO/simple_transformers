from abc import ABC
import numpy as np
from pathlib import Path
import torch as th
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM

from simple_transformers.modality_processors import MODALITY_PROCESSORS
from simple_transformers.transformer_heads import TransformHead, ClassificationHead, TokenReconstructionHead, \
    LinearReconstructionHead, DeconvReconstructionHead
from simple_transformers.utils import _init_weights, get_config

from typing import Any, Dict, List, Tuple, Union


class TransformerMixin(object):
    def setup_heads(self, preprocessor, loss_types, for_encoder=True, **kwargs):
        heads = {}
        if ('reconstructive' in loss_types and for_encoder) or ('generative' in loss_types and not for_encoder):
            reconstruction_types = ['tok_reconst'] if (self.use_hf and for_encoder) else preprocessor.get_reconstruction_types()
            for reconst_type in reconstruction_types:
                if reconst_type == 'lin_reconst':
                    assert 'state_shape' in kwargs
                    out_size = np.prod(kwargs['state_shape'])
                    heads['lin_reconst'] = LinearReconstructionHead(self.config, out_size=out_size, **kwargs)
                elif reconst_type == 'tok_reconst':
                    if self.use_hf and for_encoder and hasattr(self.encoder, 'lm_head'):
                        heads['tok_reconst'] = None
                    else:
                        emb_weights = preprocessor.get_embedding_weights()
                        heads['tok_reconst'] = TokenReconstructionHead(self.config, emb_weights)
                elif reconst_type == 'deconv_reconst':
                    cnn_in_shape, cnn_out_shape, cnn_flat_shape = preprocessor.get_encoder_intermediate_shapes()
                    heads['deconv_reconst'] = DeconvReconstructionHead(self.config, cnn_in_shape, cnn_out_shape, cnn_flat_shape)
        if 'predictive' in loss_types and for_encoder:
            heads['cls'] = ClassificationHead(self.config, kwargs['num_classes'])
        if 'contrastive' in loss_types and for_encoder:
            # Value taken from: https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPConfig
            self.logit_scale_init_value = 2.6592
            self.logit_scale = nn.Parameter(th.ones([]) * self.logit_scale_init_value)
            heads['cont'] = TransformHead(self.config)
        return nn.ModuleDict(heads)

    def check_modalities(self, modalities):
        for modality in modalities:
            if modality not in MODALITY_PROCESSORS:
                raise NotImplementedError(f'A processor for input modality "{modality}" has not yet been implemented.'
                                          f'The following modalities do have processors: {MODALITY_PROCESSORS.keys()}.')

    def _init_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def logits_to_output(self, logits):
        _, action_ids = th.max(logits, dim=-1)
        # TODO have tokenize decode option
        return action_ids

    def generate_square_subsequent_mask(self, sz: int) -> th.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return th.triu(th.full((sz, sz), float('-inf'), device=self.config.device), diagonal=1)

    def save(self, name, tag):
        Path(self.base_dir / 'models').mkdir(parents=True, exist_ok=True)
        th.save(self.state_dict(), self.base_dir / 'models' / f'{name}_{tag}')

    def load(self, name, tag):
        self.load_state_dict(th.load(self.base_dir / 'models' / f'{name}_{tag}'), map_location=self.config.device)


class ModalityEncoder(nn.Module, TransformerMixin):
    def __init__(self, modality: str,  loss_types: List, base_dir: str, **kwargs):
        """
        Encode most modalities using a transformer
        :param modality: A string defining kind of modality to encode. e.g. "text" or "images".
                         See MODALITY_PROCESSORS in simple_transformers.modality_processors for all options
        :param num_classes: If int, creates a classification transformer head
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.check_modalities([modality])
        self.config = get_config()
        self.base_dir = base_dir
        self.use_hf = False
        self.preprocessor = MODALITY_PROCESSORS[modality](self.config, **kwargs)
        self.encoder_layers = nn.TransformerEncoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         activation='gelu', batch_first=True)
        self.encoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, self.config.n_layers, self.encoder_norm)

        self.encoder_heads = self.setup_heads(self.preprocessor, loss_types, **kwargs)

        self._init_parameters()
        self.to(self.config.device)

    def forward(self, model_input: Any, attention_mask: Union[np.array, None] = None) -> Dict[str, Any]:
        """
        :param model_input: input modality. Input type depends on the modality being encoded (e.g. for text, use str)
        :param attention_mask: Attention mask on input modality. None if attention mask is dealt using modalit processor
        :return: Depending on if num classes has been defined and return_full_output, return either:
                 1) Full output of the transformer (all token embds). Shape = (batch size, seq len, d_model)
                 2) Single embedding for the full input (using the CLS token emb). Shape = (batch size, d_model)
                 3) If num classes has been defined, return logits. Shape = (batch size, num_classes)
        """
        embeddings, attention_mask = self.preprocessor(model_input, attention_mask)
        output = self.transformer_encoder(embeddings, src_key_padding_mask=attention_mask.bool())
        return_embs = {'none': output}
        for key in self.encoder_heads:
            if key == 'cls':
                return_embs[key] = self.encoder_heads[key](output[:, 0, :])
            return_embs[key] = self.encoder_heads[key](output)
        return return_embs


class ModalityDecoder(nn.Module, TransformerMixin):
    def __init__(self, modality: str, loss_types: List, base_dir: str, **kwargs):
        """
        Decoder Transformer. Modalities can be any modality from MODALITY_PROCESSORS
        Based on: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        :param output_modality: A string defining kind of modality to decode. e.g. "text" or "images".
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.check_modalities([modality])
        self.config = get_config()
        self.base_dir = base_dir
        self.use_hf = False
        self.preprocessor = MODALITY_PROCESSORS[modality](self.config, **kwargs)
        # Decoder
        self.decoder_layers = nn.TransformerDecoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         activation='gelu', batch_first=True)
        self.decoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, self.config.n_layers, self.decoder_norm)

        self.decoder_heads = self.setup_heads(self.preprocessor, loss_types, for_encoder=False, **kwargs)

        self._init_parameters()
        self.to(self.config.device)

    def forward(self, encoder_output: Any, tgt_input: Any,
                tgt_att_mask: Union[np.array, None] = None) -> Dict[str, th.Tensor]:
        """
        :param src_input: encoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        :param tgt_input: decoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        Returns
        """
        # TODO currently always uses teacher forcing. There should be an option for iteratively decoding to be used in testing

        tgt_embeddings, tgt_attention_mask = self.preprocessor(tgt_input, tgt_att_mask)
        batch_size, tgt_seq_len, d_model = tgt_embeddings.shape
        causal_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask=causal_mask,
                                      tgt_key_padding_mask=tgt_attention_mask.bool())

        return_embs = {'none': decoder_output}
        for key in self.decoder_heads:
            if key == 'cls':
                return_embs[key] = self.decoder_heads[key](decoder_output[:, 0, :])
            return_embs[key] = self.decoder_heads[key](decoder_output)

        # Output should be shape (batch size, seq len, d_model).
        return return_embs


class ModalityEncoderDecoder(nn.Module, TransformerMixin):
    def __init__(self, input_modality: str, output_modality: str, loss_types: List, base_dir: str,
                 pretrained_model: Union[Tuple[Any, Any]]=(None, None), **kwargs):
        """
        Encoder Decoder Transfomer. Modalities can be any modality from MODALITY_PROCESSORS
        Based on: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        :param input_modality: A string defining kind of modality to encode. e.g. "text" or "images".
                               See MODALITY_PROCESSORS in simple_transformers.modality_processors for all options
        :param output_modality: A string defining kind of modality to encode. e.g. "text" or "images".
                                See MODALITY_PROCESSORS in simple_transformers.modality_processors for all options
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.check_modalities([input_modality, output_modality])
        self.config = get_config()
        self.base_dir = base_dir

        pretrained_model_name, pretrained_model_tag = pretrained_model
        self.use_hf = pretrained_model_tag and ('hf' in pretrained_model_tag)

        # Encoder
        if input_modality == 'text' and self.use_hf:
            if 'pretokenized' in kwargs and kwargs['pretokenized']:
                self.preprocessor = None
                self.pretokenized = True
            else:
                self.preprocessor = RobertaTokenizer.from_pretrained(pretrained_model_name)
                self.pretokenized = False
            if 'pretrained' in pretrained_model_tag:
                self.encoder = RobertaForMaskedLM.from_pretrained(pretrained_model_name)
            else:
                self.encoder = RobertaForMaskedLM(RobertaConfig().from_pretrained(pretrained_model_name))
            self.config.d_model = self.encoder.config.hidden_size
        else:
            self.preprocessor = MODALITY_PROCESSORS[input_modality](self.config, **kwargs)
            self.encoder_layers = nn.TransformerEncoderLayer(self.config.d_model, self.config.n_heads,
                                                             self.config.hidden_size, self.config.dropout_prob,
                                                             activation='gelu', batch_first=True)
            self.encoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
            self.encoder = nn.TransformerEncoder(self.encoder_layers, self.config.n_layers, self.encoder_norm)
        # Decoder
        self.output_preprocessor = MODALITY_PROCESSORS[output_modality](self.config, **kwargs)
        self.decoder_layers = nn.TransformerDecoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         activation='gelu', batch_first=True)
        self.decoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, self.config.n_layers, self.decoder_norm)

        self.encoder_heads = self.setup_heads(self.preprocessor, loss_types, for_encoder=True, **kwargs)
        self.decoder_heads = self.setup_heads(self.output_preprocessor, loss_types, for_encoder=False, **kwargs)

        if not self.use_hf and pretrained_model_name is not None:
            self.load(pretrained_model_name, pretrained_model_tag)

        self._init_parameters()
        self.to(self.config.device)

    def encode(self, src_input: Any, src_att_mask: Union[np.array, None] = None) -> Dict[str, th.Tensor]:
        if self.use_hf:
            if not self.pretokenized:
                tok_out = self.preprocessor(src_input)
                src_input, src_att_mask = tok_out['input_ids'], tok_out['attention_mask']
            src_input, src_att_mask = th.tensor(src_input, device=self.config.device, dtype=int), th.tensor(src_att_mask, device=self.config.device, dtype=int)
            output = self.encoder.roberta(input_ids=src_input, attention_mask=src_att_mask)[0]
        else:
            src_embeddings, src_att_mask = self.preprocessor(src_input, src_att_mask)
            output = self.encoder(src_embeddings, src_key_padding_mask=src_att_mask.bool())

        return_embs = {'none': output}
        for key in self.encoder_heads:
            if key == 'cls':
                return_embs[key] = self.encoder_heads[key](output[:, 0, :])
            elif key == 'tok_reconst' and self.use_hf:
                return_embs[key] = self.encoder.lm_head(output[0])
            else:
                return_embs[key] = self.encoder_heads[key](output)

        return return_embs

    def decode(self, encoder_output: th.Tensor, tgt_input: Any, tgt_att_mask: Union[np.array, None] = None) \
            -> Dict[str, th.Tensor]:
        # TODO currently always uses teacher forcing. There should be an option for iteratively decoding to be used in testing
        tgt_embeddings, tgt_attention_mask = self.output_preprocessor(tgt_input, tgt_att_mask)
        batch_size, tgt_seq_len, d_model = tgt_embeddings.shape
        causal_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        # Output should be shape (batch size, seq len, d_model).
        decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask=causal_mask,
                                      tgt_key_padding_mask=tgt_attention_mask.bool())

        return_embs = {'none': decoder_output}
        for key in self.decoder_heads:
            return_embs[key] = self.decoder_heads[key](decoder_output)

        return return_embs

    def forward(self, src_input: Any, tgt_input: Any, src_att_mask: Union[np.array, None] = None,
                tgt_att_mask: Union[np.array, None] = None) -> Tuple[Dict[str, th.Tensor], Dict[str, th.Tensor]]:
        """
        :param src_input: encoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        :param tgt_input: decoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        Returns
        """
        encoder_output = self.encode(src_input, src_att_mask)
        decoder_output = self.decode(encoder_output['none'], tgt_input, tgt_att_mask)
        return encoder_output, decoder_output