import torch
import torch.nn as nn
from sympy.strategies.core import switch

from hyper_params import HyperParams, ActivationFunction

model_file_path = '.'


class MNISTClassifier(nn.Module):
    def __init__(self, hyper_param):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hyper_param.hidden_layer) - 1):
            self.layers.append(nn.Linear(hyper_param.hidden_layer[i], hyper_param.hidden_layer[i + 1]))
        self.layers.append(nn.Linear(hyper_param.hidden_layer[-1], 10))

        self.activation = hyper_param.activation_fun.value[1]

    def forward(self, x):
        x = x.view(x.size(0), -1)  # [batch_size, 784]

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        x = self.layers[-1](x)
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
