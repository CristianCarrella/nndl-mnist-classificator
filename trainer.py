import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from hyper_params import HyperParams, NetworkHyperParams
from network import MNISTClassifier


class Trainer:

    def __init__(self, model: MNISTClassifier,
                 hyper_params: HyperParams,
                 training_ds: DataLoader,
                 validation_ds: DataLoader,
                 testing_ds: DataLoader,
                 device: str,
                 network_hyper_params: NetworkHyperParams,
                 ):
        self.model = model
        self.hyper_params = hyper_params
        self.training_ds = training_ds
        self.validation_ds = validation_ds
        self.testing_ds = testing_ds
        self.device = device
        self.network_hyper_params = network_hyper_params

        # Initialize lists to store training and validation losses
        self.train_losses = []
        self.validation_losses = []

    def batch_train(self):
        mnist_classifier = self.model
        params = self.hyper_params
        prev_val_loss = sys.maxsize
        num_of_validation_try = 0

        for e in range(self.hyper_params.epochs):
            mnist_classifier.train()

            total_train_loss = 0.0
            num_train_batches = 0

            for batch, label in self.training_ds:
                params.optimizer.zero_grad()

                train_output = mnist_classifier.forward(batch.to(self.device))
                train_loss = params.error_function(train_output, label.to(self.device))

                train_loss.backward()
                params.optimizer.step()

                total_train_loss += train_loss.item()
                num_train_batches += 1

            avg_train_loss = total_train_loss / num_train_batches
            self.train_losses.append(avg_train_loss)  # Save training loss

            total_validation_loss = 0.0
            num_val_batches = 0

            for val_batch, val_label in self.validation_ds:
                output_validation = mnist_classifier.forward(val_batch.to(self.device))
                validation_loss = params.error_function(output_validation, val_label.to(self.device))
                total_validation_loss += validation_loss.item()
                num_val_batches += 1

            avg_validation_loss = total_validation_loss / num_val_batches
            self.validation_losses.append(avg_validation_loss)  # Save validation loss

            print(f"Epoch {e + 1}/{params.epochs}, Average Training Loss: {avg_train_loss}, "
                  f"Average Validation Loss: {avg_validation_loss}")

            if num_of_validation_try > 5:
                break
            else:
                if avg_validation_loss < prev_val_loss:
                    prev_val_loss = avg_validation_loss
                    mnist_classifier.save_model()
                else:
                    num_of_validation_try += 1

        self.plot_training_graph()

    def test(self):
        self.model.eval()
        torch.no_grad()

        good_test = 0
        total_test = 0

        for images, labels in self.testing_ds:
            outputs = self.model(images.to(self.device))
            predicted_class_indices = torch.argmax(outputs, dim=1)
            good_test += (predicted_class_indices == labels.to(self.device)).sum().item()
            total_test += labels.size(0)

        accuracy = (good_test / total_test) * 100
        print(f"Test Accuracy: {good_test}/{total_test} ({accuracy:.2f}%)")

        self.plot_testing_graph(good_test, total_test)

    def plot_training_graph(self):
        if len(self.train_losses) == self.hyper_params.epochs and len(
                self.validation_losses) == self.hyper_params.epochs:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, self.hyper_params.epochs + 1), self.train_losses, label='Training Loss')
            plt.plot(range(1, self.hyper_params.epochs + 1), self.validation_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Losses\nList of neurons for each hidden layer: {self.network_hyper_params.hidden_layer} \nActivation function: {self.network_hyper_params.activation_fun.value[0]}')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot Precision, Recall, F1 Score
            # plt.figure(figsize=(10, 6))
            # plt.plot(range(1, self.hyper_params.epochs + 1), self.precisions, label='Precision')
            # plt.plot(range(1, self.hyper_params.epochs + 1), self.recalls, label='Recall')
            # plt.plot(range(1, self.hyper_params.epochs + 1), self.f1_scores, label='F1 Score')
            # plt.xlabel('Epoch')
            # plt.ylabel('Score')
            # plt.title('Precision, Recall and F1 Score During Training')
            # plt.legend()
            # plt.grid(True)
            # plt.show()
        else:
            print(
                f"Training and validation losses are not the same length. Training losses: {len(self.train_losses)}, Validation losses: {len(self.validation_losses)}")

    # def plot_testing_graph(self, good_test, total_test):
    #     accuracy = (good_test / total_test) * 100
    #     # Plot testing accuracy
    #     plt.figure(figsize=(6, 4))
    #     plt.bar(['Test Accuracy'], [accuracy])
    #     plt.ylabel('Accuracy (%)')
    #     plt.title('Testing Accuracy')
    #     plt.ylim(0, 100)
    #     plt.show()
