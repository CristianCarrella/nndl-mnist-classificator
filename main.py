import torch
from torch.nn import CrossEntropyLoss
from torchvision.transforms import v2
import torchvision.transforms as transforms

from dataset import CustomDataset
from hyper_params import HyperParams, ActivationFunction, NetworkHyperParams
from trainer import Trainer
from network import MNISTClassifier

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

if __name__ == '__main__':
    newtork_hyper_params = NetworkHyperParams(
        hidden_layer = [784, 512, 256, 64, 32],
        activation_fun = ActivationFunction.RELU
    )
    model = MNISTClassifier(newtork_hyper_params).to(device)

    hyper_params = HyperParams(
        epochs=1,  # number of epochs
        error_function=CrossEntropyLoss(),  # error function
        is_batch=True,
        optimizer=torch.optim.Rprop(model.parameters()),  # optimizer
    )

    print(f"Epochs {hyper_params.epochs}")
    print(f"Error fun: {hyper_params.error_function}")
    print(f"Opt: {hyper_params.optimizer}")
    print(f"Is batch: {hyper_params.is_batch}")
    print(f"hidden_layer: {newtork_hyper_params.hidden_layer}")
    print(f"Activation Function: {newtork_hyper_params.activation_fun.value[0]}")

    custom_dataset = CustomDataset(
        root='./data',
        transforms=transforms.Compose([
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    training_set, validation_set = custom_dataset.get_training_set(
        is_batch=hyper_params.is_batch,
        validation_percentage=0.2,
    )

    test_set = custom_dataset.get_test_set()

    print(f"Training set length: {len(training_set.dataset)}")
    print(f"Validation set length: {len(validation_set.dataset)}")
    print(f"Test set length: {len(test_set.dataset)}")

    trainer = Trainer(
        model=model,  # model to train
        hyper_params=hyper_params,
        training_ds=training_set,
        validation_ds=validation_set,
        testing_ds=test_set,
        device=device
    )

    if not model.load_model():
        print(f"Is GPU available: {torch.cuda.is_available()}")
        trainer.batch_train()
    else:
        trainer.test()
