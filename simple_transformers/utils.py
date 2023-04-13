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


def get_config():
    import yaml

    with open("simple_transformers/config.yaml") as config_file:
        config = yaml.safe_load(config_file)
    config = SimpleNamespace(**config)
    config.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    return config
