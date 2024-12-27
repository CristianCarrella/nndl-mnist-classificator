from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer


class HyperParams:
    def __init__(self, epochs: int, error_function: CrossEntropyLoss, optimizer: Optimizer, is_batch: bool, batch_size = None):
        self.batch_size = batch_size
        self.error_function = error_function
        self.epochs = epochs
        self.optimizer = optimizer
        self.is_batch = is_batch
