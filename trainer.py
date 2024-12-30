import math
import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from hyper_params import HyperParams, NetworkHyperParams
from network import MNISTClassifier


# Trainer manages the training, validation, and testing processes for the MNIST classification model. It includes:
#
# - Training: Trains the model over multiple epochs, calculating losses and accuracies for both training and validation sets. Includes early stopping based on validation loss and saves the model if improvement is observed.
# - Testing (with Confusion Matrix): Evaluates the model on the test dataset, calculates overall accuracy, and generates a confusion matrix saved as an image.
# - Visualization (chart, log) : Plots and saves graphs of training/validation loss and accuracy over epochs, with clear labeling of model configurations.


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

        self.train_losses = []
        self.validation_losses = []

        self.train_accuracies = []
        self.validation_accuracies = []

        self.good_test = 0
        self.total_test = 0

    def batch_train(self):
        mnist_classifier = self.model
        params = self.hyper_params
        prev_val_loss = sys.maxsize
        early_stop_counter = 0

        for e in range(self.hyper_params.epochs):
            mnist_classifier.train()

            total_train_loss = 0.0
            num_train_batches = 0
            correct_train = 0
            total_train = 0

            for batch, label in self.training_ds:
                params.optimizer.zero_grad()

                train_output = mnist_classifier.forward(batch.to(self.device))
                train_loss = params.error_function(train_output, label.to(self.device))

                train_loss.backward()
                params.optimizer.step()

                total_train_loss += train_loss.item()
                num_train_batches += 1

                # Calculate training accuracy
                predicted = torch.argmax(train_output, dim=1)
                correct_train += (predicted == label.to(self.device)).sum().item()
                total_train += label.size(0)

            avg_train_loss = total_train_loss / num_train_batches
            self.train_losses.append(avg_train_loss)
            train_accuracy = correct_train / total_train
            self.train_accuracies.append(train_accuracy)

            total_validation_loss = 0.0
            num_val_batches = 0
            correct_val = 0
            total_val = 0

            for val_batch, val_label in self.validation_ds:
                output_validation = mnist_classifier.forward(val_batch.to(self.device))
                validation_loss = params.error_function(output_validation, val_label.to(self.device))
                total_validation_loss += validation_loss.item()
                num_val_batches += 1

                predicted_val = torch.argmax(output_validation, dim=1)
                correct_val += (predicted_val == val_label.to(self.device)).sum().item()
                total_val += val_label.size(0)

            avg_validation_loss = total_validation_loss / num_val_batches
            self.validation_losses.append(avg_validation_loss)
            val_accuracy = correct_val / total_val
            self.validation_accuracies.append(val_accuracy)

            print(f"Epoch {e + 1}/{params.epochs}, Average Training Loss: {avg_train_loss:.6f}, "
                  f"Average Validation Loss: {avg_validation_loss:.6f}, Training Accuracy: {train_accuracy:.4f}, "
                  f"Validation Accuracy: {val_accuracy:.4f}")

            # Early Stopping:
            if avg_validation_loss < prev_val_loss:
                prev_val_loss = avg_validation_loss
                mnist_classifier.save_model()  # Save model only when there's improvement
                early_stop_counter = 0  # Reset counter when loss improves
            else:
                early_stop_counter += 1  # Increment counter if no improvement

            # Stop training if no improvement in 5 consecutive epochs
            if early_stop_counter >= 5:
                print("Early stopping triggered: No improvement in validation loss for 5 epochs.")
                break

        self.plot_training_graph()

    def test(self):
        self.model.eval()
        torch.no_grad()

        self.good_test = 0
        self.total_test = 0
        all_labels = []
        all_predictions = []

        for images, labels in self.testing_ds:
            outputs = self.model(images.to(self.device))
            predicted_class_indices = torch.argmax(outputs, dim=1)
            self.good_test += (predicted_class_indices == labels.to(self.device)).sum().item()
            self.total_test += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_class_indices.cpu().numpy())

        accuracy = (self.good_test / self.total_test) * 100
        print(f"Test Accuracy: {self.good_test}/{self.total_test} ({accuracy:.2f}%)")

        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        # Plot and save confusion matrix
        activation_name = str(self.network_hyper_params.activation_fun[0]).replace('ActivationFunction.',
                                                                                   '')
        num_layers = len(self.network_hyper_params.hidden_layer) - 1

        os.makedirs("confusion_matrices", exist_ok=True)

        filename = f"confusion_matrices/{num_layers}_layers_{activation_name}_confusion_matrix.png"
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix\n{num_layers} Layers, {activation_name} Activation")
        plt.savefig(filename)
        print(f"Confusion matrix saved as {filename}")
        plt.close()

    def plot_training_graph(self):
        if len(self.train_losses) > 0 and len(self.validation_losses) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
            plt.plot(range(1, len(self.validation_losses) + 1), self.validation_losses, label='Validation Loss')
            plt.plot(range(1, len(self.train_accuracies) + 1), self.train_accuracies, label='Training Accuracy')
            plt.plot(range(1, len(self.validation_accuracies) + 1), self.validation_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Loss / Accuracy')

            activation_name = str(self.network_hyper_params.activation_fun[0]).replace('ActivationFunction.', '')

            plt.title(
                f'Training and Validation Metrics \nList of neurons for each hidden layer: {self.network_hyper_params.hidden_layer} \nActivation function: {activation_name}')
            plt.legend()
            plt.grid(True)

            os.makedirs("plots", exist_ok=True)
            filename = f"plots/{len(self.network_hyper_params.hidden_layer) - 1}_layers_{activation_name}_epoch.png"
            plt.savefig(filename)
            print(f"Ho salvato file {filename}")
            plt.close()
        else:
            print("No data available to create the training graph.")
