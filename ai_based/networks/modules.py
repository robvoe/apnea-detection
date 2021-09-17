from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

import ai_based.utilities.mixins as mixins


class ExtendedLinear(nn.Sequential):
    """Encapsulates a single linear layer, optionally including dropout, batch normalization and an activation."""
    @dataclass
    class LayerConfig(mixins.DictLike):
        in_features: int
        out_features: int
        use_batchnorm: bool
        dropout: Optional[float] = None
        activation_fn: Optional[nn.Module] = None

    def __init__(self, config: LayerConfig):
        super(ExtendedLinear, self).__init__()

        self.add_module("linear_layer", nn.Linear(config.in_features, config.out_features))
        if config.use_batchnorm:
            self.add_module("batch_norm", nn.BatchNorm1d(config.out_features))
        if config.activation_fn is not None:
            self.add_module("activation_fn", config.activation_fn)
        if config.dropout is not None:
            self.add_module("dropout", nn.Dropout(config.dropout))


class ExtendedConv1D(nn.Sequential):
    @dataclass
    class LayerConfig(mixins.DictLike):
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int
        use_batchnorm: bool
        pool_factor: Optional[int] = None
        dropout: Optional[float] = None
        activation_fn: Optional[nn.Module] = nn.ReLU()

    def __init__(self, config: LayerConfig):
        super(ExtendedConv1D, self).__init__()

        self.add_module("conv", nn.Conv1d(config.in_channels, config.out_channels, config.kernel_size, config.stride))
        if config.pool_factor is not None and config.pool_factor > 1:
            self.add_module("pool", nn.MaxPool1d(config.pool_factor, stride=config.pool_factor))
        if config.use_batchnorm:
            self.add_module("batch_norm", nn.BatchNorm1d(config.out_channels))
        if config.activation_fn is not None:
            self.add_module("activation_fn", config.activation_fn)
        if config.dropout is not None:
            self.add_module("dropout", nn.Dropout(config.dropout))

    @staticmethod
    def calc_output_feature_dimension(in_features, kernel_size, stride, pool_factor):
        out_features = (in_features - kernel_size) // stride + 1
        if pool_factor is not None:
            out_features = out_features // pool_factor
        return out_features
