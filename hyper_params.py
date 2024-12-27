from torch.nn import CrossEntropyLoss
from torch.optim import SGD


class HyperParams:
    def __init__(self, epochs: int, error_function: CrossEntropyLoss, optimizer: SGD, batch_size: int):
        self.batch_size = batch_size
        self.error_function = error_function
        self.epochs = epochs
        self.optimizer = optimizer
