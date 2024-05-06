from abc import ABC
import numpy as np
from pathlib import Path
import torch as th
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, AutoConfig, GPT2LMHeadModel, AutoTokenizer

from simple_transformers.modality_processors import MODALITY_PROCESSORS
from simple_transformers.transformer_heads import TransformHead, ClassificationHead, TokenReconstructionHead, \
    LinearReconstructionHead, DeconvReconstructionHead, ProjectionHead
from simple_transformers.utils import _init_weights, get_config

from typing import Any, Dict, List, Tuple, Union
from types import SimpleNamespace


class TransformerMixin(object):
    def setup_heads(self, preprocessor, inc_action_rec=False, **kwargs):
        self.transform_head = TransformHead(self.config)
        if inc_action_rec:
            print("WE BE DOING IT")
            self.action_rec_head = TokenReconstructionHead(self.config, preprocessor.get_embedding_weights())
            
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
        return th.triu(th.full((sz, sz), 1., device=self.config.device), diagonal=1).bool()

    def inference_decoding(self, start_seqs, att_mask, max_new_tokens, tokenizer, temperature=1, **kwargs):
        with th.no_grad():
            if isinstance(start_seqs, np.ndarray):
                start_seqs = th.tensor(start_seqs, device=self.config.device)
            if isinstance(att_mask, np.ndarray):
                att_mask = th.tensor(att_mask, device=self.config.device)
            # curr_seqs, att_mask = th.tensor(start_seqs, device=self.config.device, dtype=int), th.tensor(att_mask, device=self.config.device, dtype=int)
            curr_seqs = start_seqs
            batch_size, seq_len = start_seqs.shape
            curr_idx = 0
            # additional_padding = th.full((batch_size, max_length - seq_len), tokenizer.pad_token, device=self.config.device)
            # curr_seqs = th.cat((curr_seqs, additional_padding), dim=1)
            # curr_idxs = th.full((batch_size,), seq_len, device=self.config.device) - th.sum(curr_seqs == tokenizer.pad_token, dim=1) - 1
            dones = th.full((batch_size,), False, device=self.config.device)

            while not th.all(dones):
                logits = self.forward(curr_seqs, att_mask)[0]['tok_reconst']
                logits = logits[:, -1, :]# / temperature
                # logits = top_k_logits(logits, k=top_k)
                # log_probs = nn.functional.softmax(logits, dim=-1)

                new_tokens = th.argmax(logits, dim=1)
                dones = th.logical_or(dones, new_tokens == tokenizer.eos_token_id)

                curr_seqs = th.cat((curr_seqs, new_tokens.reshape(batch_size, 1)), dim=1)
                att_mask = th.cat( (att_mask, th.ones( (batch_size, 1), device=self.config.device) ), dim=1)
                curr_idx += 1
                if curr_idx >= max_new_tokens:
                    break

                # print(tokenizer.batch_decode(curr_seqs))
        return curr_seqs

    def save(self, name, tag):
        Path(self.base_dir / 'models').mkdir(parents=True, exist_ok=True)
        print(f'Saving model to: {self.base_dir} / models / {name}_{tag}')
        th.save(self.state_dict(), self.base_dir / 'models' / f'{name}_{tag}')

    def load(self, name, tag):
        tag = tag.strip('hf_')
        print(f'Loading model from: {self.base_dir} / models / {name}_{tag}')
        self.load_state_dict(th.load(self.base_dir / 'models' / f'{name}_{tag}', map_location=self.config.device), strict=False)


class ModalityEncoder(nn.Module, TransformerMixin):
    def __init__(self, modality: str, base_dir: str, config: SimpleNamespace=None, inc_action_mask=False, **kwargs):
        """
        Encode most modalities using a transformer
        :param modality: A string defining kind of modality to encode. e.g. "text" or "images".
                         See MODALITY_PROCESSORS in simple_transformers.modality_processors for all options
        :param num_classes: If int, creates a classification transformer head
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.check_modalities([modality])
        self.config = config or get_config()
        self.base_dir = base_dir
        self.modality = modality
        self.inc_action_mask = inc_action_mask
        self.use_hf = False
        self.preprocessor = MODALITY_PROCESSORS[modality](self.config, **kwargs)
        self.encoder_layers = nn.TransformerEncoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         activation='gelu', batch_first=True)
        self.encoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, self.config.n_layers, self.encoder_norm)

        self.encoder_heads = self.setup_heads(self.preprocessor, inc_action_rec=self.inc_action_mask, **kwargs)

        self._init_parameters()
        self.to(self.config.device)

    def forward(self, model_input: Any, attention_mask: Union[np.array, None] = None) -> Tuple[Dict[str, Any], th.Tensor]:
        """
        :param model_input: input modality. Input type depends on the modality being encoded (e.g. for text, use str)
        :param attention_mask: Attention mask on input modality. None if attention mask is dealt using modalit processor
        :return: Depending on if num classes has been defined and return_full_output, return either:
                 1) Full output of the transformer (all token embds). Shape = (batch size, seq len, d_model)
                 2) Single embedding for the full input (using the CLS token emb). Shape = (batch size, d_model)
                 3) If num classes has been defined, return logits. Shape = (batch size, num_classes)
        """
        embeddings, attention_mask = self.preprocessor(model_input, attention_mask)
        output = self.transformer_encoder(embeddings, src_key_padding_mask=(1 - attention_mask).bool())
        return_embs = {'none': output, 'transform': self.transform_head(output)}

        if self.inc_action_mask:
            return_embs['tok_reconst'] = self.action_rec_head(output)
        return return_embs, attention_mask


class ModalityDecoder(nn.Module, TransformerMixin):
    def __init__(self, modality: str, base_dir: str, config: SimpleNamespace=None, **kwargs):
        """
        Decoder Transformer. Modalities can be any modality from MODALITY_PROCESSORS
        Based on: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        :param output_modality: A string defining kind of modality to decode. e.g. "text" or "images".
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.check_modalities([modality])
        self.config = config or get_config()
        self.base_dir = base_dir
        self.use_hf = False
        self.preprocessor = MODALITY_PROCESSORS[modality](self.config, **kwargs)
        # Decoder
        self.decoder_layers = nn.TransformerEncoderLayer(self.config.d_model, self.config.n_heads,
                                                         self.config.hidden_size, self.config.dropout_prob,
                                                         activation='gelu', batch_first=True)
        self.decoder_norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.decoder = nn.TransformerEncoder(self.decoder_layers, self.config.n_layers, self.decoder_norm)

        # self.decoder_heads = self.setup_heads(self.preprocessor, **kwargs)

        emb_weights = self.preprocessor.get_embedding_weights()
        self.lm_head = TokenReconstructionHead(self.config, emb_weights)

        self._init_parameters()
        self.to(self.config.device)

    def generate_causal_mask_for_left_padding(self, batch_size, seq_len, attention_mask):# pad_lengths, prefix_lengths):
        # -> batch_size x seq_len x seq_len.
        base_causal_mask = self.generate_square_subsequent_mask(seq_len).repeat(batch_size, 1, 1)
        causal_mask = th.full_like(base_causal_mask, False)

        pad_len = th.sum(attention_mask == 0, dim=1)
        # Unmasks the first pad_len tokens in each sequence since if they can't attend to themselves they become
        # NaN and mess everything else up

        for b in range(batch_size):
            causal_mask[b, pad_len[b]:, pad_len[b]:] = base_causal_mask[b, :seq_len-pad_len[b], :seq_len-pad_len[b]]

        # Repeats per attention head.
        return th.repeat_interleave(causal_mask, self.config.n_heads, dim=0).to(self.config.device)

    def forward(self, model_input: Any, attention_mask: Union[np.array, None] = None, position_ids=None) -> Tuple[Dict[str, Any], th.Tensor]:
        """
        :param src_input: encoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        :param src_input: decoder input. Input type depends on the modality being encoded (e.g. for text, use str)
        Returns
        """
        if isinstance(attention_mask, np.ndarray):
            model_input = th.tensor(model_input, device=self.config.device, dtype=int)
            attention_mask = th.tensor(attention_mask, device=self.config.device)

        using_left_pad = th.any(attention_mask[:, 0] == 0)
        if using_left_pad:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        # TODO currently always uses teacher forcing. There should be an option for iteratively decoding to be used in testing
        embeddings, attention_mask = self.preprocessor(model_input, attention_mask, position_ids=position_ids)
        batch_size, seq_len, d_model = embeddings.shape

        causal_mask = (self.generate_causal_mask_for_left_padding(batch_size, seq_len, attention_mask)
                      if using_left_pad else
                      self.generate_square_subsequent_mask(seq_len))

        output = self.decoder(embeddings,  mask=causal_mask, is_causal=False,#not using_left_pad,
                              src_key_padding_mask=(1 - attention_mask).bool())

        return_embs = {'none': output, 'tok_reconst': self.lm_head(output)}

        return return_embs, attention_mask

class HFDecoder(nn.Module, TransformerMixin):
    def __init__(self, modality: str, base_dir: str, config: SimpleNamespace=None, **kwargs):
        """
        Encode most modalities using a transformer
        :param modality: A string defining kind of modality to encode. e.g. "text" or "images".
                         See MODALITY_PROCESSORS in simple_transformers.modality_processors for all options
        :param num_classes: If int, creates a classification transformer head
        :param kwargs: kwargs that define the data. Requirements vary by dataset and modality
        """
        super().__init__()
        self.check_modalities([modality])
        self.preprocessor = None
        self.config = config or get_config()
        self.model_name = kwargs['model_name']
        self.hf_config = AutoConfig.from_pretrained(kwargs['model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model_name'], padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        self.config.d_model = self.hf_config.hidden_size
        self.config.n_layers = self.hf_config.num_hidden_layers
        self.base_dir = base_dir
        self.use_hf = True
        self.decoder = None
        self.decoder_heads = self.setup_heads(self.preprocessor, for_encoder=True, **kwargs)
        self._init_parameters()
        self.to(self.config.device)

    def forward(self, model_input: Any, attention_mask: Union[np.array, None] = None, position_ids=None) -> Tuple[Dict[str, Any], th.Tensor]:
        """
        :param model_input: input modality. Input type depends on the modality being encoded (e.g. for text, use str)
        :param attention_mask: Attention mask on input modality. None if attention mask is dealt using modalit processor
        :return: Depending on if num classes has been defined and return_full_output, return either:
                 1) Full output of the transformer (all token embds). Shape = (batch size, seq len, d_model)
                 2) Single embedding for the full input (using the CLS token emb). Shape = (batch size, d_model)
                 3) If num classes has been defined, return logits. Shape = (batch size, num_classes)
        """
        # ATTENTION MASK HERE SHOULD BE 1st where we want att and 0s elsewhere
        model_input, attention_mask = model_input.long(), attention_mask.long()
        # Output should be shape (batch size, seq len, d_model).
        output = self.decoder.transformer(model_input, attention_mask=attention_mask, position_ids=position_ids,
                                          use_cache=False)[0]
        return_embs = {'none': output, 'tok_reconst': self.decoder.lm_head(output),
                       'transform': self.transform_head(output)}
        return return_embs, attention_mask

    def inference_decoding(self, start_seqs, att_mask, max_new_tokens, tokenizer):
        if not isinstance(start_seqs, th.Tensor):
            start_seqs, att_mask = th.tensor(start_seqs, device=self.config.device, dtype=int), \
                                   th.tensor(att_mask, device=self.config.device, dtype=int)
        else:
            start_seqs, att_mask = start_seqs.to(self.config.device), att_mask.to(self.config.device)

        output = self.decoder.generate(start_seqs, attention_mask=att_mask, max_new_tokens=max_new_tokens,
                                       return_dict_in_generate=True, output_scores=True)#, output_logits=True)
        output.scores = th.softmax(th.stack(output.scores, dim=1), dim=-1)
        return output.sequences, output.scores

    def load(self, name, tag):
        print(f'Loading HuggingFace Model {self.model_name}')
        if tag == 'base_hf':
            self.decoder = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.config.device)
        else:
            tag = tag.strip('hf_')
            self.decoder = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.config.device)
            self.load_state_dict(th.load(self.base_dir / 'models' / f'{name}_{tag}', map_location=self.config.device), strict=False)




if __name__ == '__main__':
    m = HFDecoder('text', 'reconstructive', '.', **{'model_name': 'gpt2'})
    m.load('gpt2', None)
    test_input = ['I love walking', 'The first program', 'My wife is a']
    test_label = [' my dog to the park', ' people write is hello world', ' great person']
    print(m.tokenizer.bos_token)
    input = m.tokenizer(test_input, padding=True, return_tensors='pt')
    label = m.tokenizer(test_label, padding=True, return_tensors='pt')
    m.inference_decoding(input['input_ids'], input['attention_mask'], label['input_ids'].shape[1], m.tokenizer, use_hf_decoding=False)
