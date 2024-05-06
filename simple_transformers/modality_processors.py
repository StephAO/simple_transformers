from abc import ABC, abstractmethod
import math
import numpy as np
from PIL.Image import Image, fromarray
import torch as th
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPImageProcessor

from types import SimpleNamespace
from typing import Union, List, Tuple, Any

from simple_transformers.utils import CNNEncoder, CNNDecoder


# NOTE: All attention masks here follow the hugging face convention of 1s for tokens, 0s for pads
# This differs from pytorch's convention (but I use HF tokenizers, so I made this call).
# When feeding into pytorch transformer models, use 1 - att_mask

class Processor(nn.Module, ABC):
    def __init__(self, config: SimpleNamespace, **kwargs):
        """
        :param config: Config namespace to hold model parameters.
                       See simple_transformers.config.yaml and simple_transformers.utils.py
        :param kwargs: kwargs required from dataset (e.g. max trajectory length)
        """
        super().__init__()
        self.check_required_kwargs(**kwargs)
        self.config = config
        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

    @abstractmethod
    def forward(self, modality: Union[str, np.array, Image, Any], att_mask: Union[np.array, None]) -> \
                Tuple[th.Tensor, th.Tensor]:
        """
        Takes in the modality you want to process. Type and format of input will vary based on modality. For example,
        a text processor could take in a string whereas an image processor could take in a PIL.Image.
        Must return input embeddings and an attention mask to feed into a transformer.
        NOTE: Attention mask follows pytorch's format, not Hugging faces format. i.e. 0->unmasked, 1->masked
        modality: Input modality
        att_mask: Attention mask on modality. Some modality processors don't require this as they will generate it
                  themselves. In this case, pass in None
        Returns: Fully formed input tokens for a transformer (including all position/modality embeddings)
                 Attention mask for the input tokens. Both should be th.tensors on the correct device
        """
        pass

    @staticmethod
    @abstractmethod
    def required_attributes() -> dict:
        pass

    def check_required_kwargs(self, **kwargs):
        for key, value in self.required_attributes().items():
            if key not in kwargs:
                raise TypeError(f'{self.__class__.__name__} missing required keyword argument: "{key}"')
            elif not isinstance(kwargs[key], value):
                raise TypeError(f'{self.__class__.__name__} keyword argument: "{key}" must be of type {value} and not '
                                f'{type(kwargs[key])}')

    def get_embedding_weights(self) -> th.Tensor:
        raise ValueError('Currently not implemented')

    def setup_position_embeddings(self, max_pos_emb, type='sincos'):
        self.max_pos_emb = max_pos_emb
        if self.config.randomize_pos_embs:
            self.max_pos_emb *= 2
        if self.config.emb_type == 'sincos':
            position = th.arange(self.max_pos_emb).unsqueeze(1)
            div_term = th.exp(th.arange(0, self.config.d_model, 2) * (-math.log(10000.0) / self.config.d_model))
            pe = th.zeros(self.max_pos_emb, self.config.d_model)
            pe[:, 0::2] = th.sin(position * div_term)
            pe[:, 1::2] = th.cos(position * div_term)
            self.register_buffer('position_embeddings', pe)
        elif self.config.emb_type == 'learned':
            self.position_embeddings = nn.Embedding(self.max_pos_emb, self.config.d_model)
            position_ids = th.arange(start=0, end=self.max_pos_emb, step=1)
            self.register_buffer('position_ids', position_ids)
        else:
            raise NotImplementedError(f'Embedding type {self.config.emb_type} has not been implemented')
        # register buffer (these are constant tokens and NOT token embeddings)

    def get_position_embeddings(self, seq_len, position_ids=None):
        if self.config.randomize_pos_embs:
            # Based on: https://aclanthology.org/2023.acl-short.161.pdf
            emb_indices = np.random.choice(np.arange(self.max_pos_emb), size=seq_len, replace=False)
            emb_indices.sort()
        elif position_ids is None:
            emb_indices = np.arange(seq_len)
        else:
            emb_indices = position_ids
        emb_indices = th.tensor(emb_indices, device=self.config.device, dtype=int)
        if self.config.emb_type == 'sincos':
            return self.position_embeddings[emb_indices]
        elif self.config.emb_type == 'learned':
            return self.position_embeddings(emb_indices)
        else:
            raise NotImplementedError(f'Embedding type {self.config.emb_type} has not been implemented')

class TextProcessor(Processor):
    """
    Processes string text into input embeddings. Uses a pretrained tokenizer from huggingface.
    Attention mask is created by tokenizer.
    REQUIRES: 'max_text_length' kwarg.
              If pretokenizing text, then 'pretokenized', 'vocab_size' and 'pad_token_id' kwargs are also required
    Can be used for generation as well
    """
    def __init__(self, config: SimpleNamespace, **kwargs):
        super().__init__(config, **kwargs)
        if 'pretokenized' in kwargs and kwargs['pretokenized']:
            self.pretokenized = True
            vocab_size, pad_token_id = kwargs['vocab_size'], kwargs['pad_token_id']
        else:
            self.pretokenized = False
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", model_max_length=256)
            vocab_size, pad_token_id = self.tokenizer.vocab_size, self.tokenizer.pad_token_id
            raise NotImplementedError
        self.word_embeddings = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_token_id)
        self.setup_position_embeddings(kwargs['max_text_length'])
        self.to(self.config.device)

    @staticmethod
    def required_attributes() -> dict:
        return {'max_text_length': int}
    
    def get_reconstruction_types(self) -> List[str]:
        return ['tok_reconst']


    def forward(self, text: Union[List[str], np.array], att_mask: None, position_ids=None) -> Tuple[th.Tensor, th.Tensor]:
        if not self.pretokenized:
            tok_out = self.tokenizer(text, padding=True, return_tensors='pt')
            tokens = tok_out['input_ids'].to(self.config.device)
            att_mask = tok_out['attention_mask']
        else:
            if isinstance(text, np.ndarray):
                text = th.from_numpy(text)
                att_mask = th.from_numpy(att_mask)
            tokens = text.to(self.config.device).int()
            att_mask = att_mask.to(self.config.device).int()
        input_embeds = self.word_embeddings(tokens)
        batch_size, seq_len, _ = input_embeds.shape
        position_embeddings = self.get_position_embeddings(seq_len, position_ids=position_ids).expand(batch_size, seq_len, -1)

        # Scale input embeddings by sqrt of d_model. Unclear why, but seems to be standard practice
        # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod/87909#87909
        embeddings = (input_embeds * math.sqrt(self.config.d_model)) + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, att_mask

    def get_embedding_weights(self) -> th.Tensor:
        return self.word_embeddings.weight

    def decode(self, token_ids: th.Tensor) -> Union[th.Tensor, str]:
        """
        If pretokenized, return token_ids, otherwise decode using tokenizer
        """
        if self.pretokenized:
            raise ValueError('Tokens were pretokenized using a tokenizer defined elsewhere')
        return self.tokenizer.batch_decode(token_ids)


class ImageProcessor(Processor):
    """
    Processes PIL.Images or np.arrays images into input embeddings. Uses a huggingface pretrained image preprocessor.
    Assumes all images are the same size (if np.array) or made to be the same size using the preprocessor (PIL.Image).
    Since images are all the same size, forward pass creates a constant sized attention mask.
    REQUIRES: 'image_size', 'patch_size', and 'num_channels', kwargs
    """
    def __init__(self, config: SimpleNamespace, **kwargs):
        super().__init__(config, **kwargs)
        # Patching logic taken from : https://github.com/huggingface/transformers/blob/v4.27.1/src/transformers/models/vit/modeling_vit.py#L141
        self.image_size = kwargs['image_size']
        self.patch_size = kwargs['patch_size']
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        self.num_channels = kwargs['num_channels']
        self.img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.patch_embeddings = nn.Conv2d(kwargs['num_channels'], config.d_model,
                                          kernel_size=self.patch_size, stride=self.patch_size)
        self.setup_position_embeddings(self.num_patches + 1)
        self.cls_embedding = nn.Parameter(th.randn(config.d_model))
        self.to(self.config.device)

    @staticmethod
    def required_attributes() -> dict:
        return {'image_size': int, 'patch_size': int, 'num_channels': int}
    
    def get_reconstruction_types(self) -> List[str]:
        # TODO not yet implemented
        return ['deconv_reconst']

    def forward(self, images: Union[List[Image], np.array], att_mask: None) -> Tuple[th.Tensor, th.Tensor]:
        """
        :param images: If using PIL.Images, then preprocess using clip image preprocessing. This includes normalization
                       and resizing to image size 224. Otherwise, condition directly on provided np.array
        """
        if isinstance(images, np.ndarray):
            pixel_values = th.tensor(images).to(self.config.device).float()
        else:
            pixel_values = self.img_processor(images, return_tensors='pt')['pixel_values'].to(self.config.device)
        batch_size, num_channels, height, width = pixel_values.shape
        # patch_embeddings returns: batch_size, d_model, image_size // patch_size, image_size // patch_size
        # flatten returns: batch_size, d_model, num_patches
        # transpose returns: batch_size, num_patches, d_model
        input_embeds = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        # Add cls token to the start of set of tokens
        input_embeds = th.cat([self.cls_embedding.expand(batch_size, 1, -1), input_embeds], dim=1)
        position_embeddings = self.get_position_embeddings(input_embeds.shape[1])
        embeddings = (input_embeds * math.sqrt(self.config.d_model)) + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # image is always the same size, so no need for padding --> attention mask is all zeros
        return embeddings, th.ones(batch_size, self.num_patches + 1, device=self.config.device)


class ActionProcessor(Processor):
    """
    Processes discrete actions into input embeddings.
    Attention mask must be passed in.
    REQUIRES: 'num_actions', 'max_seq_length' kwargs.
    Can be used for generation as well
    """
    def __init__(self, config: SimpleNamespace, **kwargs):
        super().__init__(config, **kwargs)
        # +1 for SOS/CLS token
        if 'num_actions' in kwargs:
            self.action_embeddings = nn.Embedding(kwargs['num_actions'] + 1, config.d_model)
            cls_id = th.tensor(kwargs['num_actions'])
            self.register_buffer('cls_id', cls_id)
            self.action_enc_mode = 'emb'
        elif 'action_dim' in kwargs:
            self.action_embeddings = nn.Linear(kwargs['action_dim'], config.d_model)
            self.cls_embedding = nn.Parameter(th.randn(config.d_model))
            self.action_enc_mode = 'lin'
        else:
            raise ValueError('ActionProcessor requires either num_actions or action_dim')
        self.setup_position_embeddings(kwargs['max_seq_length'] + 1)
        self.to(self.config.device)

    @staticmethod
    def required_attributes() -> dict:
        return {'max_seq_length': int}
    
    def get_reconstruction_types(self) -> List[str]:
        return ['tok_reconst']

    def forward(self, actions: np.array, att_mask: np.array) -> Tuple[th.Tensor, th.Tensor]:
        # Trajectories should already be padded at this point
        batch_size, traj_length = actions.shape
        actions = th.from_numpy(actions).to(self.config.device).int()
        att_mask = th.from_numpy(att_mask).to(self.config.device).int()
        # Add cls token to the start of each traj
        if self.action_enc_mode == 'emb':
            actions = th.cat([self.cls_id.expand(batch_size, 1).to(self.config.device), actions], dim=1)
        att_mask = th.cat([th.ones(batch_size, 1, device=self.config.device), att_mask], dim=1)
        # Embed
        input_embeds = self.action_embeddings(actions)
        if self.action_enc_mode == 'lin':
            input_embeds = th.cat([self.cls_embedding.expand(batch_size, 1, -1), input_embeds], dim=1)
        position_embeddings = self.get_position_embeddings(input_embeds.shape[1])
        embeddings = (input_embeds * math.sqrt(self.config.d_model)) + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, att_mask

    def get_embedding_weights(self) -> th.Tensor:
        return self.action_embeddings.weight

class TrajectoryProcessor(Processor):
    """
    Processes a grid world representation into input embeddings.
    Attention mask must be passed in.
    REQUIRES: 'state_shape', 'max_actions', and 'traj_type' kwargs
    Trajectory type can be one of: 'states', 'states_actions', 'fl_states', 'fl_states_actions'
    where fl means only first and last states are used
    """
    def __init__(self, config: SimpleNamespace, **kwargs):
        super().__init__(config, **kwargs)
        input_size = np.prod(kwargs['state_shape'])
        self.state_embeddings = nn.Linear(input_size, config.d_model)
        # + 1 for MASK token
        self.action_embeddings = nn.Embedding(kwargs['num_diff_actions'] + 1, config.d_model)
        self.traj_type = kwargs['traj_type']
        # +1 for CLS token
        self.setup_position_embeddings(kwargs['max_seq_length'] + 1)
        self.cls_embedding = nn.Parameter(th.randn(config.d_model))
        self.to(self.config.device)

    @staticmethod
    def required_attributes() -> dict:
        return {'state_shape': List, 'max_seq_length': int, 'traj_type': str, 'num_diff_actions': int}

    def forward(self, traj: np.array, att_mask: np.array) -> Tuple[th.Tensor, th.Tensor]:
        # Trajectories should already be padded at this point
        states, actions = traj
        states_att_mask, actions_att_mask = att_mask
        batch_size, traj_length, *_ = states.shape

        # Embed states and actions
        state_embeds = self.state_embeddings(th.flatten(states, start_dim=2).float())
        if 'actions' in self.traj_type:
            action_embeds = self.action_embeddings(actions.int())
        # Join states and actions and attention_masks
        if self.traj_type == 'states' or self.traj_type == 'fl_states':
            # Extracting first and last state is done in dataset before padding
            input_embeds = state_embeds
            att_mask = states_att_mask
        elif self.traj_type == 'states_actions':
            # Interleaves states and actions
            # Remove final state to match length of actions
            traj_length -= 1
            input_embeds = th.stack((state_embeds[:, :-1], action_embeds), dim=2).reshape(batch_size, 2 * traj_length, -1)
            att_mask = th.stack((states_att_mask[:, :-1], actions_att_mask), dim=2).reshape(batch_size, 2 * traj_length)
            # Re-add final state
            input_embeds = th.cat([input_embeds, state_embeds[:, -1:]], dim=1)
            att_mask = th.cat([att_mask, states_att_mask[:, -1:]], dim=1)
        elif self.traj_type == 'fl_states_actions':
            # Extracting first and last state is done in dataset before padding.
            # First state will be at index 0 and last state will be at index 1
            input_embeds = th.cat([state_embeds[:, 0].unsqueeze(1), action_embeds, state_embeds[:, 1].unsqueeze(1)], dim=1)
            att_mask = th.cat([states_att_mask[:, 0].unsqueeze(1), actions_att_mask, states_att_mask[:, 1].unsqueeze(1)], dim=1)
        else:
            raise ValueError(f'Trajectory type {self.traj_type} not recognized')

        # Add cls token to the start of each traj
        input_embeds = th.cat([self.cls_embedding.expand(batch_size, 1, -1), input_embeds], dim=1)
        att_mask = th.cat([th.ones(batch_size, 1, device=self.config.device), att_mask.int()], dim=1)
        # Add position embeddings
        position_embeddings = self.get_position_embeddings(input_embeds.shape[1])
        # Combine, normalize, dropout
        embeddings = (input_embeds * math.sqrt(self.config.d_model)) + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, att_mask



MODALITY_PROCESSORS = {
    'text': TextProcessor,
    'images': ImageProcessor,
    'actions': ActionProcessor,
    'trajs': TrajectoryProcessor,
}
