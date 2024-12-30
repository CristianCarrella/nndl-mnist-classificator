import itertools

import torch
from torch.nn import CrossEntropyLoss
from torchvision.transforms import v2
import torchvision.transforms as transforms

from dataset import CustomDataset
from hyper_params import HyperParams, ActivationFunction, NetworkHyperParams
from logger import log_results
from trainer import Trainer, iterations
from network import MNISTClassifier

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def generate_layer_combinations(hidden_layers_list):
    functions = [af.value[1] for af in ActivationFunction]
    results = []
    for l in hidden_layers_list:
        activation_combinations = list(itertools.product(functions, repeat=len(hidden_layers) - 1))
        for activation_combo in activation_combinations:
            results.append((hidden_layers, activation_combo))
    return results


if __name__ == '__main__':
    hidden_layers = [
        [784, 512, 32],
        [784, 512, 256, 128, 32],  # Aggiunta di layer intermedi
        [784, 400, 100, 25],  # Progressione più dolce
        [784, 512, 128],  # Configurazione più semplice
        [784, 256, 64, 32],  # Riduzione più drastica
        [784, 412, 50],  # Simile alla seconda tua ma più robusta
        [784, 600, 300, 100],  # Layer più grandi
        [784, 1024, 512, 256],  # Configurazione molto ampia
        [784, 300, 150, 50, 25],  # Progressione lineare più lenta
        [784, 600, 25],  # Riduzione drastica dopo un inizio ampio
    ]

    activation_functions = [
        # Per configurazioni da 5
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT,
            ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT,
            ActivationFunction.RELU, ActivationFunction.LEAKY_RELU
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT,
            ActivationFunction.LEAKY_RELU, ActivationFunction.HYPERBOLIC_TANGENT
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT,
            ActivationFunction.LEAKY_RELU, ActivationFunction.LEAKY_RELU
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.LEAKY_RELU, ActivationFunction.HYPERBOLIC_TANGENT,
            ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.LEAKY_RELU, ActivationFunction.HYPERBOLIC_TANGENT,
            ActivationFunction.RELU, ActivationFunction.LEAKY_RELU
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.LEAKY_RELU, ActivationFunction.HYPERBOLIC_TANGENT,
            ActivationFunction.LEAKY_RELU, ActivationFunction.HYPERBOLIC_TANGENT
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.LEAKY_RELU, ActivationFunction.HYPERBOLIC_TANGENT,
            ActivationFunction.LEAKY_RELU, ActivationFunction.LEAKY_RELU
        ],
        [
            ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU,
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.HYPERBOLIC_TANGENT
        ],
        [
            ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU,
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.LEAKY_RELU
        ],

        # Per configurazioni da 4
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU, ActivationFunction.LEAKY_RELU
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.LEAKY_RELU, ActivationFunction.HYPERBOLIC_TANGENT
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.LEAKY_RELU, ActivationFunction.LEAKY_RELU
        ],
        [
            ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.HYPERBOLIC_TANGENT
        ],
        [
            ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.LEAKY_RELU
        ],

        # Per configurazioni da 3
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.RELU
        ],
        [
            ActivationFunction.HYPERBOLIC_TANGENT, ActivationFunction.LEAKY_RELU
        ],
        [
            ActivationFunction.RELU, ActivationFunction.HYPERBOLIC_TANGENT
        ],
        [
            ActivationFunction.RELU, ActivationFunction.LEAKY_RELU
        ],

    ]


    custom_dataset = CustomDataset(
        root='./data',
        transforms=transforms.Compose([
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    training_set, validation_set = custom_dataset.get_training_set(
        is_batch=True,
        validation_percentage=0.2,
    )

    test_set = custom_dataset.get_test_set()
    epochs = 100
    print(f"Training set length: {len(training_set.dataset)}")
    print(f"Validation set length: {len(validation_set.dataset)}")
    print(f"Test set length: {len(test_set.dataset)}")

    param_combinations = itertools.product(hidden_layers, activation_functions)
    for combination in param_combinations:
        if combination[0].__len__() - 1 == combination[1].__len__():  # combinations are valid only if activation functions are 1 less of hidden layers
            log_results({"id": iterations, "type": "params", "hidden_layer": combination[0],
                         "activation_functions": [el.value[0] for el in combination[1]], "max_epochs": epochs,
                         "error_function": "CrossEntropyLoss"})

            network_hyper_params = NetworkHyperParams(
                hidden_layer=combination[0],
                activation_fun=combination[1]  # has to be hidden_layer len - 1
            )

            model = MNISTClassifier(network_hyper_params).to(device)

            hyper_params = HyperParams(
                patience=5,
                epochs=epochs,  # number of epochs
                error_function=CrossEntropyLoss(),  # error function
                is_batch=True,
                optimizer=torch.optim.Rprop(model.parameters()),  # optimizer
            )

            print(combination)
            print(f"Epochs {hyper_params.epochs}")
            print(f"Error fun: {hyper_params.error_function}")
            print(f"Opt: {hyper_params.optimizer}")
            print(f"Is batch: {hyper_params.is_batch}")
            print(f"hidden_layer: {network_hyper_params.hidden_layer}")
            print(f"Activation Function: {network_hyper_params.activation_fun}")
            print(model)

            trainer = Trainer(
                model=model,  # model to train
                hyper_params=hyper_params,
                training_ds=training_set,
                validation_ds=validation_set,
                testing_ds=test_set,
                device=device,
                network_hyper_params=network_hyper_params
            )

            trainer.train()
            trainer.test()

            iterations += 1
            # if not model.load_model():
            #     print(f"Is GPU available: {torch.cuda.is_available()}")
            # else:
            #     trainer.test()
