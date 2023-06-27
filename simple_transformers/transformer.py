from abc import ABC
import numpy as np
import torch as th
import torch.nn as nn
from simple_transformers.modality_processors import MODALITY_PROCESSORS
from simple_transformers.transformer_heads import TransformHead, ClassificationHead, TokenReconstructionHead, ReconstructionHead
from simple_transformers.utils import _init_weights, get_config

from typing import Any, Dict, List, Tuple, Union


class TransformerMixin(object):
    def setup_heads(self, **kwargs):
        self.heads = {}
        if True: # TODO
            self.heads['trans'] = TransformHead(self.config)
        if 'reconst' in kwargs:
            assert 'state_size' in kwargs
            self.heads['reconst'] = ReconstructionHead(self.config, out_size=kwargs['state_size'], **kwargs)
        if 'tok_reconst' in kwargs: # TODO
            self.heads['tok_reconst'] = TokenReconstructionHead(self.config, self.preprocessor.get_embedding_weights())
        if 'num_classes' in kwargs:
            self.heads['cls'] = ClassificationHead(self.config, kwargs['num_classes'])
        self.heads = nn.ModuleDict(self.heads)

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


class ModalityEncoder(nn.Module, TransformerMixin):
    def __init__(self, modality: str, **kwargs):
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
        self.preprocessor = MODALITY_PROCESSORS[modality](self.config, **kwargs)
        self.encoder_layers = nn.TransformerEncoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         batch_first=True)
        self.encoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, self.config.n_layers, self.encoder_norm)

        self.setup_heads(**kwargs)

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
        for key in self.heads:
            if key == 'cls':
                return_embs[key] = self.heads[key](output[:, 0, :])
            return_embs[key] = self.heads[key](output)
        return return_embs

class ModalityDecoder(nn.Module, TransformerMixin):
    def __init__(self, modality: str, **kwargs):
        """
        Decoder Transformer. Modalities can be any modality from MODALITY_PROCESSORS
        Based on: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        :param output_modality: A string defining kind of modality to decode. e.g. "text" or "images".
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.check_modalities([modality])
        self.config = get_config()
        self.preprocessor = MODALITY_PROCESSORS[modality](self.config, **kwargs)
        # Decoder
        self.decoder_layers = nn.TransformerDecoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         batch_first=True)
        self.decoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, self.config.n_layers, self.decoder_norm)

        self.setup_heads(**kwargs)

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
        batch_size, tgt_seq_len, emb_dim = tgt_embeddings.shape
        causal_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask=causal_mask,
                                      tgt_key_padding_mask=tgt_attention_mask.bool())

        return_embs = {'none': decoder_output}
        for key in self.heads:
            if key == 'cls':
                return_embs[key] = self.heads[key](decoder_output[:, 0, :])
            return_embs[key] = self.heads[key](decoder_output)

        # Output should be shape (batch size, seq len, d_model).
        return return_embs


class ModalityEncoderDecoder(nn.Module, TransformerMixin):
    def __init__(self, input_modality: str, output_modality: str, **kwargs):
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
        self.preprocessor = MODALITY_PROCESSORS[input_modality](self.config, **kwargs)
        self.output_preprocessor = MODALITY_PROCESSORS[output_modality](self.config, **kwargs)
        # Encoder
        self.encoder_layers = nn.TransformerEncoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         batch_first=True)
        self.encoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, self.config.n_layers, self.encoder_norm)
        # Decoder
        self.decoder_layers = nn.TransformerDecoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         batch_first=True)
        self.decoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, self.config.n_layers, self.decoder_norm)

        self.setup_heads(**kwargs)

        self._init_parameters()
        self.to(self.config.device)

    def encode(self, src_input: Any, src_att_mask: Union[np.array, None] = None) -> Dict[str, th.Tensor]:
        src_embeddings, src_att_mask = self.preprocessor(src_input, src_att_mask)
        output = self.encoder(src_embeddings, src_key_padding_mask=src_att_mask.bool())

        return_embs = {'none': output}
        for key in self.heads:
            if key == 'cls':
                return_embs[key] = self.heads[key](output[:, 0, :])
            return_embs[key] = self.heads[key](output)

        return return_embs

    def decode(self, encoder_output: th.Tensor, tgt_input: Any, tgt_att_mask: Union[np.array, None] = None) \
               -> Dict[str, th.Tensor]:
        # TODO currently always uses teacher forcing. There should be an option for iteratively decoding to be used in testing
        tgt_embeddings, tgt_attention_mask = self.output_preprocessor(tgt_input, tgt_att_mask)
        batch_size, tgt_seq_len, emb_dim = tgt_embeddings.shape
        causal_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        # Output should be shape (batch size, seq len, d_model).
        decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask=causal_mask,
                                      tgt_key_padding_mask=tgt_attention_mask.bool())

        return_embs = {'none': decoder_output}
        for key in self.heads:
            if key == 'cls':
                return_embs[key] = self.heads[key](decoder_output[:, 0, :])
            return_embs[key] = self.heads[key](decoder_output)

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