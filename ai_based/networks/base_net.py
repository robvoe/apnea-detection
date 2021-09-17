from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch.nn as nn

import ai_based.utilities.mixins as mixins


class BaseNet(nn.Module, ABC):
    @dataclass
    class Config(mixins.DictLike):
        """Base configuration container for all networks."""
        input_tensor_shape: Tuple
        output_tensor_shape: Tuple

    def __init__(self, config: Config):
        super(BaseNet, self).__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        pass

    def test(self, data):
        self.eval()
        results = self.forward(data)
        self.train()
        return results

    @property
    def device(self):
        """
        Return the device the model is stored on. It is assumed that all weights are always residing on a single device.
        """
        return next(self.parameters()).device

    def num_parameters(self):
        return sum([t.nelement() for _, t in self.state_dict().items()])
