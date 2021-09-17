from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch.nn as nn

from .base_net import BaseNet
from .modules import ExtendedLinear
import ai_based.utilities.mixins as mixins


class MLP(BaseNet):
    """A modern and generic MLP featuring batch normalization and dropout."""
    @dataclass
    class Config(BaseNet.Config):
        @dataclass
        class HiddenLayerConfig(mixins.DictLike):
            out_features: int
            use_batchnorm: bool
            dropout: Optional[float] = None
            activation_fn: Optional[nn.Module] = nn.ReLU()

        hidden_layer_configs: Iterable[HiddenLayerConfig]
        last_layer_dropout: Optional[float]
        last_layer_use_batchnorm: bool

    def __init__(self, config: Config):
        super(MLP, self).__init__(config)
        self.layers = nn.Sequential()
        current_n_features = int(np.prod(config.input_tensor_shape))
        for i, layer_config in enumerate(config.hidden_layer_configs):
            extended_linear_config = ExtendedLinear.LayerConfig(in_features=current_n_features,
                                                                out_features=layer_config.out_features,
                                                                use_batchnorm=layer_config.use_batchnorm,
                                                                dropout=layer_config.dropout,
                                                                activation_fn=layer_config.activation_fn)
            self.layers.add_module(f"layer_{i}", ExtendedLinear(extended_linear_config))
            current_n_features = layer_config.out_features

        last_layer_config = ExtendedLinear.LayerConfig(in_features=current_n_features,
                                                       out_features=int(np.prod(config.output_tensor_shape)),
                                                       use_batchnorm=config.last_layer_use_batchnorm,
                                                       dropout=config.last_layer_dropout,
                                                       activation_fn=None)
        self.layers.add_module("layer_last", ExtendedLinear(last_layer_config))

    def forward(self, x):
        x = nn.Flatten(-2)(x)
        x = self.layers(x)
        x = nn.Unflatten(-1, self.config.output_tensor_shape)(x)
        return x
