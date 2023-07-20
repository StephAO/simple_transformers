import numpy as np
import torch as th
import torch.nn as nn
from types import SimpleNamespace

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.1, 0.1)
    elif hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight)


def get_output_shape(model, image_dim):
    return model(th.rand(*(image_dim))).data.shape[1:]


class CNNEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super().__init__()

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
        )
        self.intermediate_shape = get_output_shape(self.cnn, [1, *input_shape])
        self.flatten = nn.Flatten()
        self.flattened_shape = np.prod(self.intermediate_shape)
        # Linear
        self.linear = nn.Linear(self.flattened_shape, hidden_dim)

    def forward(self, z):
        is_seq = (len(z.shape) == 5)
        if is_seq:
            batch_size, seq_len, *rem_shape = z.shape
            z = z.reshape(batch_size * seq_len, *rem_shape)
        z = self.linear(self.flatten(self.cnn(z)))
        if is_seq:
            z = z.reshape(batch_size, seq_len, -1)
        return z

class CNNDecoder(nn.Module):

    def __init__(self, hidden_dim, cnn_output_shape, cnn_flattened_shape):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, cnn_flattened_shape)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=cnn_output_shape)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, z):
        return th.sigmoid(self.deconv(self.unflatten(self.linear(z))))


def get_config():
    import yaml
    from pkg_resources import resource_filename

    with open(resource_filename('simple_transformers', 'config.yaml')) as config_file:
        config = yaml.safe_load(config_file)
    config = SimpleNamespace(**config)
    config.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    return config
