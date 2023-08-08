import torch as th
import torch.nn as nn

from simple_transformers.utils import CNNDecoder


class TransformHead(nn.Module):
    def __init__(self, config, input_size=None, **kwargs):
        super(TransformHead, self).__init__()
        input_size = input_size if input_size else config.d_model
        self.head = nn.Sequential(
            nn.Linear(input_size, config.emb_dim),
            nn.GELU(),
            nn.LayerNorm(config.emb_dim, eps=1e-12)
        )

    def forward(self, hidden_states):
        return self.head(hidden_states)


class LinearReconstructionHead(nn.Module):
    def __init__(self, config, out_size=None, act_fn=nn.Tanh, out_scale=1, **kwargs):
        super(LinearReconstructionHead, self).__init__()
        assert out_size is not None
        self.transform_head = TransformHead(config)
        self.decoder = nn.Sequential(nn.Linear(config.emb_dim, out_size),
                                     act_fn())
        self.out_scale = out_scale

    def forward(self, hidden_states):
        hidden_states = self.transform_head(hidden_states)
        hidden_states = self.decoder(hidden_states) * self.out_scale
        return hidden_states

class DeconvReconstructionHead(nn.Module):
    def __init__(self, config, cnn_in_shape, enc_output_shape, enc_flattened_shape, out_scale=1):
        super(DeconvReconstructionHead, self).__init__()
        self.transform_head = TransformHead(config)
        self.decoder = CNNDecoder(config.emb_dim, cnn_in_shape=cnn_in_shape,
                                  cnn_output_shape=enc_output_shape, cnn_flattened_shape=enc_flattened_shape)
        self.out_scale = out_scale

    def forward(self, hidden_states):
        hidden_states = self.transform_head(hidden_states)
        hidden_states = self.decoder(hidden_states) * self.out_scale
        return hidden_states

class TokenReconstructionHead(nn.Module):
    def __init__(self, config, embedding_weights, input_size=None, **kwargs):
        super(TokenReconstructionHead, self).__init__()
        input_size = input_size if input_size else config.d_model
        self.transform_head = TransformHead(config, input_size=input_size)
        self.decoder = nn.Linear(embedding_weights.size(1),
                                 embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = embedding_weights
        self.bias = nn.Parameter(th.zeros(embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform_head(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class ClassificationHead(nn.Module):
    def __init__(self, config, num_classes, input_size=None, **kwargs):
        super(ClassificationHead, self).__init__()
        input_size = input_size if input_size else config.d_model
        self.head = nn.Sequential(
            TransformHead(config, input_size=input_size),
            nn.Linear(config.emb_dim, num_classes)
        )

    def forward(self, hidden_states):
        return self.head(hidden_states)
