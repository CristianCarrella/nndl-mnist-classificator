from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer


class HyperParams:
    def __init__(self, epochs: int, error_function: CrossEntropyLoss, optimizer: Optimizer, batch_size: int):
        self.batch_size = batch_size
        self.error_function = error_function
        self.epochs = epochs
        self.optimizer = optimizer
