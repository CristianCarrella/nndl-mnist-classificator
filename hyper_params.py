from enum import Enum
from typing import List

from torch.nn import CrossEntropyLoss
import torch.nn as nn
from torch.optim import Optimizer


class ActivationFunction(Enum):
    RELU = ("relu", nn.ReLU())
    LEAKY_RELU = ("leakyRelu", nn.LeakyReLU())
    HYPERBOLIC_TANGENT = ("hyperbolicTangent", nn.Tanh())


class HyperParams:
    def __init__(self, epochs: int, error_function: CrossEntropyLoss, optimizer: Optimizer, is_batch: bool, batch_size=None):
        self.batch_size = batch_size
        self.error_function = error_function
        self.epochs = epochs
        self.is_batch = is_batch
        self.optimizer = optimizer


class NetworkHyperParams:
    def __init__(self, hidden_layer: List[int], activation_fun: ActivationFunction):
        self.hidden_layer = hidden_layer
        self.activation_fun = activation_fun
