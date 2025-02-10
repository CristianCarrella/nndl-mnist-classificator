import random
from math import floor

import torch
from torch.nn import CrossEntropyLoss
from torchvision.transforms import v2
import torchvision.transforms as transforms
import os
from dataset import CustomDataset
from hyper_params import HyperParams, ActivationFunction, NetworkHyperParams
from trainer import Trainer
from network import MNISTClassifier

# The main trains and tests different neural network model for MNIST dataset for various network configurations.
# A Cartesian product is computed between different activation functions and network architectures (number of layers and neurons).
# For each combination:
# - The model is initialized with the specified parameters.
# - Training, validation, and test datasets are loaded and prepared.
# - Training is performed using an iterative process, saving metrics per epoch.
# - Training details, such as losses and accuracies, along with test results, are saved to a log file.
# The number of neurons per layer is fixed for time constraints.


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def random_search(n_iter=100):
    configurations = []

    for _ in range(n_iter):
        num_layers = random.randint(1, 4)  # Numero di layer tra 1 e 5
        hidden_units = [random.randint(16, 512) for _ in range(num_layers)]  # Numero di neuroni per ogni layer
        hidden_units.insert(0, 784)
        hidden_units.sort(reverse=True)
        activation_functions = [random.choice(list(ActivationFunction)) for _ in
                                range(num_layers)]  # Funzione di attivazione per ogni layer

        config = {
            "num_layers": num_layers,
            "hidden_units": hidden_units,
            "activation_functions": [activation.value for activation in activation_functions]
        }
        configurations.append(config)

    return configurations


def random_layer(layer_num_neurons, input_layer_size, min_layer_size):
    single_config = [784]
    input_layer_size = input_layer_size - min_layer_size
    decrement = floor(input_layer_size / (layer_num_neurons + 1))
    max_range = input_layer_size - 1
    min_range = input_layer_size - decrement
    for _ in range(layer_num_neurons):
        random_number = random.randrange(min_range, max_range)
        max_range = min_range - 1
        min_range = min_range - decrement - 1
        single_config.append(random_number)
    return single_config


if __name__ == '__main__':

    con = random_search(10)
    activation_functions = [
        ActivationFunction.RELU,
        ActivationFunction.LEAKY_RELU,
        ActivationFunction.HYPERBOLIC_TANGENT,
    ]

    neuron_configs = {}
    for i in range(1, 6):
        neuron_configs[i] = random_layer(i, 784, 32)

    print(neuron_configs)


    for config_len, layers in neuron_configs.items():
        for activation_function in activation_functions:
            network_hyper_params = NetworkHyperParams(
                hidden_layer=layers,
                activation_fun=[activation_function] * (len(layers) - 1)
            )

            model = MNISTClassifier(network_hyper_params).to(device)
            print(f"\n\nTraining Model with Layers {layers} and Activation Function {activation_function}")

            hyper_params = HyperParams(
                epochs=100,
                error_function=CrossEntropyLoss(),
                is_batch=True,
                optimizer=torch.optim.Rprop(model.parameters()),
            )

            custom_dataset = CustomDataset(
                root='./data',
                transforms=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            )

            training_set, validation_set = custom_dataset.get_training_set(
                is_batch=hyper_params.is_batch,
                validation_percentage=0.2,
            )

            test_set = custom_dataset.get_test_set()

            trainer = Trainer(
                model=model,
                hyper_params=hyper_params,
                training_ds=training_set,
                validation_ds=validation_set,
                testing_ds=test_set,
                device=device,
                network_hyper_params=network_hyper_params
            )

            trainer.batch_train()

            log_file_name = f"training_detailsGeneralizationLoss/{config_len}_layers_{activation_function}.txt"
            os.makedirs("training_detailsGeneralizationLoss", exist_ok=True)

            with open(log_file_name, 'w') as log_file:
                log_file.write(
                    f"Training completed for model with {layers} layers and activation function {activation_function}\n")
                log_file.write(f"Epochs: {hyper_params.epochs}\n")
                log_file.write(f"Optimizer: {hyper_params.optimizer}\n")
                log_file.write(f"Hidden layers: {layers}\n")
                log_file.write("Training and Validation Metrics per Epoch:\n")

                for e in range(hyper_params.epochs):
                    if e < len(trainer.train_losses) and e < len(trainer.validation_losses):
                        log_file.write(
                            f"Epoch {e + 1}: "
                            f"Training Loss: {trainer.train_losses[e]:.6f}, Validation Loss: {trainer.validation_losses[e]:.6f}, "
                            f"Training Accuracy: {trainer.train_accuracies[e]:.4f}, Validation Accuracy: {trainer.validation_accuracies[e]:.4f}\n"
                        )
                    else:
                        log_file.write(f"Epoch {e + 1}: No data available for this epoch due to early stopping\n")

                trainer.test()

                good_test = trainer.good_test
                total_test = trainer.total_test
                accuracy = (good_test / total_test) * 100
                log_file.write(f"\nTest Results:\n")
                log_file.write(f"Accuracy: {good_test}/{total_test} ({accuracy:.2f}%)\n")
                log_file.write("Confusion matrix displayed in output.\n")

            print(f"Training details and test results saved to {log_file_name}")
