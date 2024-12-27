from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets


class CustomDataset(Dataset):
    """
      Classe personalizzata per il dataset MNIST estesa con funzionalità:
       - per caricare il dataset MNIST
       - suddividere in set di addestramento e validazione
       - ottenere il set di test.

      Attributi:
          full_test_set: Il dataset di test MNIST.
          get_training_set: Si ottengono il training_set ed il validation_set
      """

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

    def get_training_set(self, batch_size=32, validation_percentage=0.1):
        train_size = int((1 - validation_percentage) * len(self.__full_training_set))
        val_size = len(self.__full_training_set) - train_size

        train_set, val_set = random_split(self.__full_training_set, [train_size, val_size])

        data_loader_train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        data_loader_val_set = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        return data_loader_train_set, data_loader_val_set

    def get_test_set(self, batch_size=32):
        return DataLoader(self.__full_test_set, batch_size=batch_size, shuffle=True)
