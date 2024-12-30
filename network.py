import torch
import torch.nn as nn

from hyper_params import NetworkHyperParams

# This module defines the MNISTClassifier class, a neural network model for classifying MNIST images. Key functionalities include:
#
# - Initialization: Dynamically constructs the layers of the network based on the specified hyperparameters, including hidden layers and activation functions.
# - Forward Pass: Implements the forward propagation logic to process input images through the layers and produce class predictions.
# - Save and Load: Provides methods to save the model's state to a file and load it later for inference or further training.

model_file_path = '.'


class MNISTClassifier(nn.Module):
    def __init__(self, hyper_param: NetworkHyperParams):
        super().__init__()
        self.layers = []
        self.acts = []
        for fun in hyper_param.activation_fun:
            self.acts.append(fun.value[1])

        for i in range(len(hyper_param.hidden_layer) - 1):
            self.layers.append(nn.Linear(hyper_param.hidden_layer[i], hyper_param.hidden_layer[i + 1]))
            self.add_module(f"layer{i}", self.layers[i])
            self.add_module(f"act{i}", self.acts[i])

        self.output = nn.Linear(hyper_param.hidden_layer[-1], 10)
        self.soft_max = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # [batch_size, 784]
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x))
        x = self.output(x)
        #x = self.soft_max(self.output(x))
        return x

    def save_model(self):
        torch.save(self.state_dict(), f"{model_file_path}/model")
        print("model saved")

    def load_model(self):
        try:
            self.load_state_dict(torch.load(f"{model_file_path}/model", weights_only=True))
            self.eval()
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"No model saved found {e}")
            return False
