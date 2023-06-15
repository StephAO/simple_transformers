import numpy as np
import torch as th
import torch.nn as nn
from simple_transformers.modality_processors import ENCODER_MODALITY_PROCESSORS, DECODER_MODALITY_PROCESSORS
from simple_transformers.transformer_heads import TransformHead, ClassificationHead, ReconstructionHead
from simple_transformers.utils import _init_weights, get_config

from typing import Any, Dict, List, Tuple, Union


class ModalityEncoder(nn.Module):
    def __init__(self, modality: str, **kwargs):
        """
        Encode most modalities using a transformer
        :param modality: A string defining kind of modality to encode. e.g. "text" or "images".
                         See ENCODER_MODALITY_PROCESSORS in simple_transformers.modality_processors for all options
        :param num_classes: If int, creates a classification transformer head
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.config = get_config()
        if modality not in ENCODER_MODALITY_PROCESSORS:
            raise NotImplementedError(f'A processor for input modality "{modality}" has not yet been implemented.'
                                      f'The following modalities do have processors: {ENCODER_MODALITY_PROCESSORS.keys()}.')
        self.preprocessor = ENCODER_MODALITY_PROCESSORS[modality](self.config, **kwargs)
        self.encoder_layers = nn.TransformerEncoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         batch_first=True)
        self.encoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, self.config.n_layers, self.encoder_norm)

        self.heads = {}
        if True: # TODO
            self.heads['trans'] = TransformHead(self.config)
        if modality == 'text': # TODO
            self.heads['reconst'] = ReconstructionHead(self.config, self.preprocessor.get_embedding_weights())
        if 'num_classes' in kwargs:
            self.heads['cls'] = ClassificationHead(self.config, kwargs['num_classes'])

        self.heads = nn.ModuleDict(self.heads)

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

    def reconstruct_input_from_embeddings(self, embeddings: th.Tensor) -> Tuple[th.Tensor, Any]:
        logits = self.preprocessor.output_embeddings_to_logits(embeddings)
        output = self.preprocessor.logits_to_output(logits)
        return logits, output

    def output_embeddings_to_logits(self, embs):
        logits = embs @ th.transpose(self.action_embeddings.weight, 0, 1)
        return logits

    def logits_to_output(self, logits):
        _, action_ids = th.max(logits, dim=-1)
        return action_ids

    def _init_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class ModalityDecoder(nn.Module):
    def __init__(self, modality: str, **kwargs):
        """
        Decoder Transformer. modality processor must implement output_embeddings_to_logits and logits_to_output.
        Based on: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        :param output_modality: A string defining kind of modality to decode. e.g. "text" or "images".
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.config = get_config()
        if modality not in DECODER_MODALITY_PROCESSORS:
            raise NotImplementedError(f'A processor for input modality "{modality}" has not yet been implemented.'
                                      f'The following modalities do have processors: {DECODER_MODALITY_PROCESSORS.keys()}.')
        self.output_preprocessor = DECODER_MODALITY_PROCESSORS[modality](self.config, **kwargs)
        # Decoder
        self.decoder_layers = nn.TransformerDecoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         batch_first=True)
        self.decoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.decoder = nn.TransformerDecoder(self.decoder_layers, self.config.n_layers, self.decoder_norm)

        self._init_parameters()
        self.to(self.config.device)

    def forward(self, encoder_output: Any, tgt_input: Any,
                tgt_att_mask: Union[np.array, None] = None) -> Tuple[th.Tensor, th.Tensor, List[Any]]:
        """
        :param src_input: encoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        :param tgt_input: decoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        Returns
        """
        tgt_embeddings, tgt_attention_mask = self.output_preprocessor(tgt_input, tgt_att_mask)
        batch_size, tgt_seq_len, emb_dim = tgt_embeddings.shape
        causal_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask=causal_mask,
                                      tgt_key_padding_mask=tgt_attention_mask.bool())

        # Output should be shape (batch size, seq len, d_model).
        logits = self.output_preprocessor.output_embeddings_to_logits(decoder_output)
        generated_output = self.output_preprocessor.logits_to_output(logits)
        return decoder_output, logits, generated_output

    def _init_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int, device='cpu') -> th.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        print()
        return th.triu(th.full((sz, sz), float('-inf'), device=self.config.device), diagonal=1)


class ModalityEncoderDecoder(nn.Module):
    def __init__(self, input_modality: str, output_modality: str, **kwargs):
        """
        Encoder Decoder Transfomer. Input modality can be any modality from ENCODER_MODALITY_PROCESSORS.
        Output modality can be any modality from DECODER_MODALITY_PROCESSORS, whose processors implement
        output_embeddings_to_logits and logits_to_output.
        Based on: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        :param input_modality: A string defining kind of modality to encode. e.g. "text" or "images".
                               See ENCODER_MODALITY_PROCESSORS in simple_transformers.modality_processors for all options
        :param output_modality: A string defining kind of modality to encode. e.g. "text" or "images".
                                See DECODER_MODALITY_PROCESSORS in simple_transformers.modality_processors for all options
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.config = get_config()
        if input_modality not in ENCODER_MODALITY_PROCESSORS:
            raise NotImplementedError(f'A processor for input modality "{input_modality}" has not yet been implemented.'
                                      f'The following modalities do have processors: {ENCODER_MODALITY_PROCESSORS.keys()}.')
        if output_modality not in DECODER_MODALITY_PROCESSORS:
            raise NotImplementedError(f'A processor for output modality "{output_modality}" has not yet been implemented.'
                                      f'The following modalities do have processors: {DECODER_MODALITY_PROCESSORS.keys()}.')
        self.input_preprocessor = ENCODER_MODALITY_PROCESSORS[input_modality](self.config, **kwargs)
        self.output_preprocessor = DECODER_MODALITY_PROCESSORS[output_modality](self.config, **kwargs)
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

        self._init_parameters()
        self.to(self.config.device)

    def encode(self, src_input: Any, src_att_mask: Union[np.array, None] = None) -> th.Tensor:
        src_embeddings, src_att_mask = self.input_preprocessor(src_input, src_att_mask)
        output = self.encoder(src_embeddings, src_key_padding_mask=src_att_mask.bool())
        return output

    def decode(self, encoder_output: th.Tensor, tgt_input: Any, tgt_att_mask: Union[np.array, None] = None) \
               -> Tuple[th.Tensor, th.Tensor, List[Any]]:
        tgt_embeddings, tgt_attention_mask = self.output_preprocessor(tgt_input, tgt_att_mask)
        batch_size, tgt_seq_len, emb_dim = tgt_embeddings.shape
        causal_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask=causal_mask,
                                      tgt_key_padding_mask=tgt_attention_mask.bool())

        # Output should be shape (batch size, seq len, d_model).
        logits = self.output_preprocessor.output_embeddings_to_logits(decoder_output)
        generated_output = self.output_preprocessor.logits_to_output(logits)
        return decoder_output, logits, generated_output

    def forward(self, src_input: Any, tgt_input: Any, src_att_mask: Union[np.array, None] = None,
                tgt_att_mask: Union[np.array, None] = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, List[Any]]:
        """
        :param src_input: encoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        :param tgt_input: decoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        Returns
        """
        encoder_output = self.encode(src_input, src_att_mask)
        decoder_output, logits, generated_output = self.decode(encoder_output, tgt_input, tgt_att_mask)
        return encoder_output, decoder_output, logits, generated_output

    def _init_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> th.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return th.triu(th.full((sz, sz), float('-inf'), device=self.config.device), diagonal=1)
