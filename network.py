import torch
import torch.nn as nn

model_file_path = '.'


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # image pre-processing
        x = x.squeeze(0)  # remove the channel (not necessary for non conv networks)
        x = x.view(-1, 28 * 28)  # make flat the images [batch_size, 784]

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
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
