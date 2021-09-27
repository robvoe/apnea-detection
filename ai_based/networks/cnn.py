from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np
import torch

from .base_net import BaseNet
from .modules import ExtendedConv1D, ExtendedLinear
from .mlp import MLP
import ai_based.utilities.mixins as mixins


class Cnn1D(BaseNet):
    """
    A convolutional neural network for processing signals received from an array of ultrasound sensors.
    """
    @dataclass
    class Config(BaseNet.Config):
        @dataclass
        class EncoderLayerConfig(mixins.DictLike):
            out_channels: int
            kernel_size: int
            stride: int
            use_batchnorm: bool
            pool_factor: Optional[int] = None
            dropout: Optional[float] = None
            activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU()

        encoder_layer_configs: Iterable[EncoderLayerConfig]
        hidden_dense_layer_configs: Iterable[MLP.Config.HiddenLayerConfig]
        last_layer_dropout: Optional[float]
        last_layer_use_batchnorm: bool

    def __init__(self, config: Config):
        super(Cnn1D, self).__init__(config)

        self.encoder = torch.nn.Sequential()
        current_n_channels = config.input_tensor_shape[0]
        for i, layer_config in enumerate(config.encoder_layer_configs):
            extended_conv1d_config = ExtendedConv1D.LayerConfig(in_channels=current_n_channels,
                                                                out_channels=layer_config.out_channels,
                                                                kernel_size=layer_config.kernel_size,
                                                                stride=layer_config.stride,
                                                                use_batchnorm=layer_config.use_batchnorm,
                                                                pool_factor=layer_config.pool_factor,
                                                                dropout=layer_config.dropout,
                                                                activation_fn=layer_config.activation_fn)
            self.encoder.add_module(f"conv_layer_{i}", ExtendedConv1D(extended_conv1d_config))
            current_n_channels = layer_config.out_channels

        verification_input = torch.zeros(1, *config.input_tensor_shape, dtype=torch.float32)
        try:
            self.encoder.eval()
            encoder_output = self.encoder(verification_input)
            self.encoder.train()
        except (ValueError, RuntimeError) as e:
            raise ValueError(f"There is a problem with the configuration of the encoder.") from e

        # Now, let's add a few dense layers
        self.decoder = torch.nn.Sequential()
        current_n_features = int(np.prod(encoder_output.shape[1:]))
        for i, layer_config in enumerate(config.hidden_dense_layer_configs):
            extended_linear_config = ExtendedLinear.LayerConfig(in_features=current_n_features,
                                                                out_features=layer_config.out_features,
                                                                use_batchnorm=layer_config.use_batchnorm,
                                                                dropout=layer_config.dropout,
                                                                activation_fn=layer_config.activation_fn)
            self.decoder.add_module(f"hidden_dense_layer_{i}", ExtendedLinear(extended_linear_config))
            current_n_features = layer_config.out_features

        # Last layer
        last_layer_config = ExtendedLinear.LayerConfig(in_features=current_n_features,
                                                       out_features=int(np.prod(config.output_tensor_shape)),
                                                       use_batchnorm=config.last_layer_use_batchnorm,
                                                       dropout=config.last_layer_dropout,
                                                       activation_fn=None)
        self.decoder.add_module("layer_last", ExtendedLinear(last_layer_config))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x.reshape(x.shape[0], -1))
        x = x.reshape(x.shape[0], *self.config.output_tensor_shape)
        return x

    @staticmethod
    def calc_all_intermediate_feature_dimensions(n_features, encoder_layer_configs):
        print(f"Layer 0: {n_features}")
        for i, layer_config in enumerate(encoder_layer_configs, 1):
            n_features = ExtendedConv1D.calc_output_feature_dimension(n_features, layer_config.kernel_size,
                                                                      layer_config.stride, layer_config.pool_factor)
            print(f"Layer {i}: {n_features}")
