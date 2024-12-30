from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets


class CustomDataset(Dataset):

    # Class that loads the MNIST dataset from PyTorch and:
    #  - Splits it into training and validation sets (80%-20%)
    #  - Retrieves the test set.
    #  MNIST contains 60,000 images for training and 10,000 for testing.

    def __init__(self, root, transforms=None, target_transform=None):
        self.__full_training_set = datasets.MNIST(
            root=root,
            train=True,
            transform=transforms,
            target_transform=target_transform,
            download=True
        )

        self.__full_test_set = datasets.MNIST(
            root=root,
            train=False,
            transform=transforms,
            target_transform=target_transform,
            download=True
        )

    def __len__(self):
        return len(self.__full_training_set)

    def __getitem__(self, idx):
        image, label = self.__full_training_set[idx]

        return image, label

    def get_training_set(self, is_batch=True, batch_size = 1, validation_percentage=0.1):
        train_batch_size = batch_size
        val_batch_size = batch_size

        train_size = int((1 - validation_percentage) * len(self.__full_training_set))
        val_size = len(self.__full_training_set) - train_size
        if is_batch:
            train_batch_size = train_size
            val_batch_size = val_size

        train_set, val_set = random_split(self.__full_training_set, [train_size, val_size])

        data_loader_train_set = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
        data_loader_val_set = DataLoader(val_set, batch_size=val_batch_size, shuffle=True)
        return data_loader_train_set, data_loader_val_set

    def get_test_set(self):
        return DataLoader(self.__full_test_set, shuffle=True)
