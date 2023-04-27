import numpy as np
import torch as th
import torch.nn as nn
from simple_transformers.modality_processors import MODALITY_PROCESSORS
from simple_transformers.utils import _init_weights, get_config

from typing import Any, Union, List, Tuple


class ModalityEncoder(nn.Module):
    def __init__(self, modality: str, **kwargs):
        """
        Encode most modalities using a transformer
        :param modality: A string defining kind of modality to encode. e.g. "text" or "images".
                         See MODALITY_PROCESSORS in simple_transformers.modality_processors for all options
        :param num_classes: int or None. If int, creates a linear layer to output class logits
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.config = get_config()
        if modality not in MODALITY_PROCESSORS:
            raise NotImplementedError(f'A processor for {modality} has yet to be implemented.'
                                      f'The following modalities do have processors: {MODALITY_PROCESSORS.keys()}.')
        self.preprocessor = MODALITY_PROCESSORS[modality](self.config, **kwargs)
        self.encoder_layers = nn.TransformerEncoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, self.config.n_layers)
        self.num_classes = None
        if 'num_classes' in kwargs:
            self.num_classes = kwargs['num_classes']
            self.classifier = nn.Linear(self.config.d_model, self.num_classes)
        self.apply(_init_weights)
        self.to(self.config.device)

    def forward(self, model_input: Any, attention_mask: Union[np.array, None] = None,
                return_full_output=False) -> th.Tensor:
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
        if return_full_output:
            return output
        # Output should be shape (batch size, seq len, d_model). Take first token (cls token) for each to return
        mod_emb = output[:, 0, :]
        # Normalize
        mod_emb = mod_emb / mod_emb.norm(p=2, dim=-1, keepdim=True)
        if self.num_classes is None:
            return mod_emb
        logits = self.classifier(mod_emb)
        return logits


class ModalityEncoderDecoder(nn.Module):
    def __init__(self, input_modality: str, output_modality: str, **kwargs):
        """
        Encoder Decoder Transfomer. Input modality can any valid modality from MODALITY_PROCESSORS.
        Currently, only text processor has required functions for decoder.
        :param input_modality: A string defining kind of modality to encode. e.g. "text" or "images".
                               See MODALITY_PROCESSORS in simple_transformers.modality_processors for all options
        :param output_modality: A string defining kind of modality to encode. e.g. "text" or "images".
                                Currenlty, only "text" works
        :param num_classes: int or None. If int, creates a linear layer to output class logits
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.config = get_config()
        if input_modality not in MODALITY_PROCESSORS:
            raise NotImplementedError(f'A processor for {input_modality} has yet to be implemented.'
                                      f'The following modalities do have processors: {MODALITY_PROCESSORS.keys()}.')
        if output_modality not in MODALITY_PROCESSORS:
            raise NotImplementedError(f'A processor for {output_modality} has yet to be implemented.'
                                      f'The following modalities do have processors: {MODALITY_PROCESSORS.keys()}.')
        self.input_preprocessor = MODALITY_PROCESSORS[input_modality](self.config, **kwargs)
        self.output_preprocessor = MODALITY_PROCESSORS[output_modality](self.config, **kwargs)
        self.transformer = nn.Transformer(self.config.d_model, self.config.n_heads, self.config.n_layers,
                                          self.config.n_layers, self.config.hidden_size,
                                          dropout=self.config.dropout_prob, batch_first=True)
        self.apply(_init_weights)
        self.to(self.config.device)

    def forward(self, src_input: Any, tgt_input: Any, src_att_mask: Union[np.array, None] = None,
                tgt_att_mask: Union[np.array, None] = None) -> Tuple[th.Tensor, List[Any]]:
        """
        :param src_input: encoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        :param tgt_input: decoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        Returns
        """
        src_embeddings, src_attention_mask = self.input_preprocessor(src_input, src_att_mask)
        tgt_embeddings, tgt_attention_mask = self.output_preprocessor(tgt_input, tgt_att_mask)
        batch_size, tgt_seq_len, emb_dim = tgt_embeddings.shape
        causal_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len)
        output = self.transformer(src_embeddings, tgt_embeddings, tgt_mask=causal_mask,
                                  src_key_padding_mask=src_attention_mask.bool(),
                                  tgt_key_padding_mask=tgt_attention_mask.bool())
        # Output should be shape (batch size, seq len, d_model).
        logits = self.output_preprocessor.output_embeddings_to_logits(output)
        generated_output = self.output_preprocessor.logits_to_output(logits)
        return logits, generated_output
