from abc import ABC, abstractmethod
import math
import numpy as np
from PIL.Image import Image, fromarray
import torch as th
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPImageProcessor

from types import SimpleNamespace
from typing import Union, List, Tuple, Any

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
        self.word_embeddings = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(kwargs['max_text_length'], config.d_model)
        position_ids = th.arange(start=0, end=kwargs['max_text_length'], step=1)
        # register buffer (these are constant tokens and NOT token embeddings)
        self.register_buffer('position_ids', position_ids)
        self.to(self.config.device)

    @staticmethod
    def required_attributes() -> dict:
        return {'max_text_length': int}


    def forward(self, text: Union[List[str], np.array], att_mask: None) -> Tuple[th.Tensor, th.Tensor]:
        if not self.pretokenized:
            tok_out = self.tokenizer(text, padding=True, return_tensors='pt')
            tokens = tok_out['input_ids'].to(self.config.device)
            att_mask = 1 - tok_out['attention_mask']
        else:
            tokens = th.from_numpy(text).to(self.config.device).int()
            att_mask = th.from_numpy(att_mask).to(self.config.device).int()
        input_embeds = self.word_embeddings(tokens)
        batch_size, seq_len, _ = input_embeds.shape
        position_embeddings = self.position_embeddings(self.position_ids[:seq_len])
        position_embeddings = position_embeddings.expand(batch_size, seq_len, -1)

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
        self.position_embeddings = nn.Embedding(self.num_patches + 1, config.d_model)  # +1 for cls token
        self.cls_embedding = nn.Parameter(th.randn(config.d_model))
        position_ids = th.arange(start=0, end=self.num_patches + 1, step=1)
        # register buffer (these are constant tokens and NOT token embeddings)
        self.register_buffer('position_ids', position_ids)
        self.to(self.config.device)

    @staticmethod
    def required_attributes() -> dict:
        return {'image_size': int, 'patch_size': int, 'num_channels': int}

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
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings = (input_embeds * math.sqrt(self.config.d_model)) + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # image is always the same size, so no need for padding --> attention mask is all zeros
        return embeddings, th.zeros(batch_size, self.num_patches + 1, device=self.config.device)


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
        self.action_embeddings = nn.Embedding(kwargs['num_actions'] + 1, config.d_model)
        self.position_embeddings = nn.Embedding(kwargs['max_seq_length'] + 1, config.d_model)
        cls_id = th.tensor(kwargs['num_actions'])
        position_ids = th.arange(start=0, end=kwargs['max_seq_length'] + 1, step=1)
        # register buffer (these are constant ids, not embeddings)
        self.register_buffer('position_ids', position_ids)
        self.register_buffer('cls_id', cls_id)
        self.to(self.config.device)

    @staticmethod
    def required_attributes() -> dict:
        return {'num_actions': int, 'max_seq_length': int}

    def forward(self, actions: np.array, att_mask: np.array) -> Tuple[th.Tensor, th.Tensor]:
        # Trajectories should already be padded at this point
        batch_size, traj_length = actions.shape
        actions = th.from_numpy(actions).to(self.config.device).int()
        att_mask = th.from_numpy(att_mask).to(self.config.device).int()
        # Add cls token to the start of each traj
        actions = th.cat([self.cls_id.expand(batch_size, 1).to(self.config.device), actions], dim=1)
        att_mask = th.cat([th.zeros(batch_size, 1, device=self.config.device), att_mask], dim=1)
        # Embed
        input_embeds = self.action_embeddings(actions)
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings = (input_embeds * math.sqrt(self.config.d_model)) + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, att_mask

    def get_embedding_weights(self) -> th.Tensor:
        return self.word_embeddings.weight

class GridStateProcessor(Processor):
    """
    Processes a grid world representation into input embeddings.
    Attention mask must be passed in.
    REQUIRES: 'state_size', 'max_seq_length' kwargs.
    """
    def __init__(self, config: SimpleNamespace, **kwargs):
        super().__init__(config, **kwargs)
        self.state_embeddings = nn.Linear(kwargs['state_size'], config.d_model)
        self.position_embeddings = nn.Embedding(kwargs['max_seq_length'] + 1, config.d_model)
        self.cls_embedding = nn.Parameter(th.randn(config.d_model))
        position_ids = th.arange(start=0, end=kwargs['max_seq_length'] + 1, step=1)
        # register buffer (these are constant tokens and NOT token embeddings)
        self.register_buffer('position_ids', position_ids)
        self.to(self.config.device)

    @staticmethod
    def required_attributes() -> dict:
        return {'state_size': int, 'max_seq_length': int}

    def forward(self, states: np.array, att_mask: np.array) -> Tuple[th.Tensor, th.Tensor]:
        # Trajectories should already be padded at this point
        print(states.shape)
        batch_size, traj_length, *_ = states.shape
        states = th.from_numpy(states).to(self.config.device)
        att_mask = th.from_numpy(att_mask).to(self.config.device).int()
        input_embeds = self.state_embeddings(th.flatten(states, start_dim=2).float())
        # Add cls token to the start of each traj
        input_embeds = th.cat([self.cls_embedding.expand(batch_size, 1, -1), input_embeds], dim=1)
        att_mask = th.cat([th.zeros(batch_size, 1, device=self.config.device), att_mask], dim=1)
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings = (input_embeds * math.sqrt(self.config.d_model)) + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, att_mask

class TrajectoryProcessor(Processor):
    """
    TODO consider stacking states & actions
    Processes a trajectory (sequence of states and actions) into input embeddings.
    Currently, states should be grid world representations.
    Attention mask must be passed in.
    REQUIRES: 'state_size', 'max_seq_length' kwargs.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        n_layers = kwargs['num_state_enc_layers'] if 'num_state_enc_layers' in kwargs else 1
        self.state_embeddings = nn.Sequential(
            *[l for i in range(n_layers) for l in
              (nn.Linear(*((kwargs['state_size'], config.d_model) if i == 0 else (config.d_model, config.d_model))),
               nn.ReLU())
              ])
        if 'decoding' in kwargs:
            self.state_decode_embeddings = nn.Sequential(
                *[l for i in range(n_layers - 1, -1, -1) for l in
                  (nn.Linear(*((config.d_model, kwargs['state_size']) if i == 0 else (config.d_model, config.d_model))),
                   nn.ReLU())
                  ])
        self.action_embeddings = nn.Embedding(kwargs['num_actions'], config.d_model)
        self.position_embeddings = nn.Embedding(kwargs['max_seq_length'] * 2 + 1, config.d_model)
        self.cls_embedding = nn.Parameter(th.randn(config.d_model))
        position_ids = th.arange(start=0, end=kwargs['max_seq_length'] * 2 + 1, step=1)
        # register buffer (these are constant tokens and NOT token embeddings)
        self.register_buffer('position_ids', position_ids)
        self.to(self.config.device)

    @staticmethod
    def required_attributes() -> dict:
        return {'state_size': int, 'max_seq_length': int}

    def forward(self, traj: Tuple[np.array, np.array], att_mask: np.array) -> Tuple[th.Tensor, th.Tensor]:
        # Trajectories should already be padded at this point
        # Move inputs to tensors on the right device
        states, actions = traj
        states_att_mask, actions_att_mask = att_mask
        batch_size, traj_length, *_ = states.shape
        states = th.from_numpy(states).to(self.config.device)
        actions = th.from_numpy(actions).to(self.config.device).int()
        states_att_mask = th.from_numpy(states_att_mask).to(self.config.device).int()
        actions_att_mask = th.from_numpy(actions_att_mask).to(self.config.device).int()
        # Create joined attention mask
        mask_sizes = th.sum(states_att_mask, dim=-1, keepdim=True) + th.sum(actions_att_mask, dim=-1, keepdim=True)
        att_mask = th.zeros((batch_size, traj_length * 2), device=self.config.device, dtype=int)
        att_mask[th.arange(0, traj_length * 2, device=self.config.device).repeat(batch_size, 1) >= mask_sizes] = 1
        # Embed states and actions
        state_embeds = self.state_embeddings(th.flatten(states, start_dim=2).float())
        action_embeds = self.action_embeddings(actions)
        # Interleaves states and actions
        input_embeds = th.stack((state_embeds, action_embeds), dim=2).reshape(batch_size, 2 * traj_length, -1)
        # Add cls token to the start of each traj
        input_embeds = th.cat([self.cls_embedding.expand(batch_size, 1, -1), input_embeds], dim=1)
        att_mask = th.cat([th.zeros(batch_size, 1, device=self.config.device), att_mask], dim=1)
        # Add position embeddings
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings = (input_embeds * math.sqrt(self.config.d_model)) + position_embeddings
        # Layernom + dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, att_mask

    def get_embedding_weights(self) -> th.Tensor:
        return self.word_embeddings.weight

    # def output_embeddings_to_logits(self, embs):
    #     # embs is a sequence of states/actions, each having to be decoded in different ways
    #     # 1. Split states and actions (skip first token whic is cls token)
    #     states_embs, action_embs = embs[:, 1::2], embs[:, 2::2]
    #     # 2. Decode each accordingly
    #     state_logits = self.state_decode_embeddings(states_embs)
    #     action_logits = action_embs @ th.transpose(self.action_embeddings.weight, 0, 1)
    #     return (state_logits, action_logits)
    #
    # def logits_to_output(self, logits):
    #     state_logits, action_logits = logits
    #     _, action_ids = th.max(action_logits, dim=-1)
    #     return (state_logits, action_ids)


class InitialStateProcessor(Processor):
    """
    Processes an initial state and mission into input embeddings
    Currently, this should be a tuple of a grid world representation and a string describing the mission
    State always has length of 1, so attention mask is calculated by increasing the length of the text
    attention mask by 1
    REQUIRES: 'state_size', 'max_text_length' kwargs.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        n_layers = kwargs['num_state_enc_layers'] if 'num_state_enc_layers' in kwargs else 1
        self.state_embeddings = nn.ModuleList(
            [nn.Linear(
                *((kwargs['state_size'], config.d_model) if i == 0 else (config.d_model, config.d_model))
             ) for i in range(n_layers)])
        self.text_processor = TextProcessor(config, **kwargs)
        self.state_type_emb = nn.Parameter(th.randn(config.d_model))
        self.text_type_emb = nn.Parameter(th.randn(config.d_model))
        self.to(self.config.device)

    @staticmethod
    def required_attributes() -> dict:
        return {'state_size': int, 'max_text_length': int}

    def forward(self, init_state: Tuple[np.array, str], att_mask: None) -> Tuple[th.Tensor, th.Tensor]:
        # Move things to tensors on the right device
        state, mission = init_state
        batch_size, grid_size, grid_size, _ = state.shape
        states = th.from_numpy(state).to(self.config.device)
        # Process text - text_embs should be shape (batch_size, text_length, d_model)
        text_embs, text_att_mask = self.text_processor(mission, None)
        # Process state
        state_embs = self.state_embeddings(th.flatten(states, start_dim=1).float())
        # Add type embeddings
        batch_size, text_len, _ = text_embs.shape
        text_embs += self.text_type_emb
        state_embs += self.state_type_emb
        # Combine text and state embeddings and include state in attention mask
        input_embeds = th.cat([state_embs.unsqueeze(1), text_embs], dim=1)
        att_mask = th.cat([th.zeros(batch_size, 1, device=self.config.device), text_att_mask], dim=1)
        # Layernom + dropout
        embeddings = self.LayerNorm(input_embeds)
        embeddings = self.dropout(embeddings)
        return embeddings, att_mask


MODALITY_PROCESSORS = {
    'text': TextProcessor,
    'images': ImageProcessor,
    'actions': ActionProcessor,
    'states': GridStateProcessor,
    'trajs': TrajectoryProcessor,
    'init_state': InitialStateProcessor
}
