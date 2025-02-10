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
                 gen_loss_threshold: float = 0.2):
        self.model = model
        self.hyper_params = hyper_params
        self.training_ds = training_ds
        self.validation_ds = validation_ds
        self.testing_ds = testing_ds
        self.device = device
        self.network_hyper_params = network_hyper_params
        self.gen_loss_threshold = gen_loss_threshold
        self.last_saved_epoch = None

        self.train_losses = []
        self.validation_losses = []
        self.train_accuracies = []
        self.validation_accuracies = []
        self.good_test = 0
        self.total_test = 0

        # Early stopping conditions
        self.patience_triggered = False
        self.generalization_triggered = False
        self.early_stopping_epochs = {}

    def batch_train(self):
        mnist_classifier = self.model
        params = self.hyper_params
        prev_val_loss = sys.maxsize
        early_stop_counter = 0
        min_val_loss = sys.maxsize

        for e in range(self.hyper_params.epochs):
            mnist_classifier.train()
            total_train_loss, correct_train, total_train = 0.0, 0, 0
            num_train_batches = 0

            for batch, label in self.training_ds:
                params.optimizer.zero_grad()
                train_output = mnist_classifier.forward(batch.to(self.device))
                train_loss = params.error_function(train_output, label.to(self.device))
                train_loss.backward()
                params.optimizer.step()

                total_train_loss += train_loss.item()
                num_train_batches += 1
                predicted = torch.argmax(train_output, dim=1)
                correct_train += (predicted == label.to(self.device)).sum().item()
                total_train += label.size(0)

            avg_train_loss = total_train_loss / num_train_batches
            self.train_losses.append(avg_train_loss)
            train_accuracy = correct_train / total_train
            self.train_accuracies.append(train_accuracy)

            total_validation_loss, correct_val, total_val = 0.0, 0, 0
            num_val_batches = 0

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

            print(f"Epoch {e + 1}/{params.epochs}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_validation_loss:.6f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            generalization_loss = 100 * ((avg_validation_loss / min_val_loss) - 1)
            print(f"Generalization Loss: {generalization_loss:.6f}")

            if avg_validation_loss < prev_val_loss:
                prev_val_loss = avg_validation_loss
                min_val_loss = avg_validation_loss
                mnist_classifier.save_model()
                early_stop_counter = 0
                self.last_saved_epoch = e
            else:
                early_stop_counter += 1

            if not self.patience_triggered and early_stop_counter >= 5 and e >= 10:
                self.patience_triggered = True
                self.early_stopping_epochs['patience'] = e
                print("Early stopping triggered: Patience")
                print(f"Test Accuracy after Patience early stop: {self.test()}")

            if not self.generalization_triggered and generalization_loss > self.gen_loss_threshold and e > 10:
                self.generalization_triggered = True
                self.early_stopping_epochs['generalization_loss'] = e
                print("Early stopping triggered: Generalization Loss")
                print(f"Test Accuracy after Generalization Loss early stop: {self.test()}")

            # Training si ferma quando entrambe le condizioni sono soddisfatte
            if self.patience_triggered and self.generalization_triggered:
                print("All early stopping conditions met. Stopping training.")
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
        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        precision_avg = np.nanmean(precision) * 100
        recall_avg = np.nanmean(recall) * 100

        os.makedirs("confusion_matricesGeneralizationLoss", exist_ok=True)
        filename = f"confusion_matricesGeneralizationLoss/{len(self.network_hyper_params.hidden_layer) - 1}_layers_{str(self.network_hyper_params.activation_fun[0]).replace('ActivationFunction.', '')}_confusion_matrix.png"
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{len(self.network_hyper_params.hidden_layer) - 1}_layers, Activation: {str(self.network_hyper_params.activation_fun[0]).replace('ActivationFunction.', '')} | Precision: {precision_avg:.2f}%, Recall: {recall_avg:.2f}%, Accuracy: {accuracy:.2f}%")
        plt.savefig(filename)
        plt.close()
        return accuracy, precision_avg, recall_avg

    def plot_training_graph(self):
        if len(self.train_losses) > 0 and len(self.validation_losses) > 0:
            test_accuracy, test_precision, test_recall = self.test()

            plt.figure(figsize=(10, 6))
            epochs = range(1, len(self.train_losses) + 1)

            plt.plot(epochs, self.train_losses, label='Training Loss', color='blue')
            plt.plot(epochs, self.validation_losses, label='Validation Loss', color='red')
            plt.plot(epochs, self.train_accuracies, label='Training Accuracy', color='green')
            plt.plot(epochs, self.validation_accuracies, label='Validation Accuracy', color='purple')

            colors = {'patience': 'yellow', 'generalization_loss': 'orange'}
            for stop_type, epoch in self.early_stopping_epochs.items():
                plt.scatter(epoch + 1, self.validation_losses[epoch], color=colors[stop_type], s=100,
                            label=f'{stop_type} Stop', edgecolors='k', zorder=3)

            if self.last_saved_epoch is not None:
                plt.scatter(self.last_saved_epoch + 1, self.validation_losses[self.last_saved_epoch], color='green',
                            s=200, edgecolors='black', marker='o', label="Last Saved Model")

            plt.xlabel('Epoch')
            plt.ylabel('Loss / Accuracy')

            num_layers = len(self.network_hyper_params.hidden_layer) - 1
            neurons_per_layer = self.network_hyper_params.hidden_layer
            activation_name = str(self.network_hyper_params.activation_fun[0]).replace('ActivationFunction.', '')

            plt.title(f'{num_layers} Layers: {neurons_per_layer} | {activation_name} Activation\n'
                      f'Test Accuracy: {test_accuracy:.2f}% | Precision: {test_precision:.2f}% | Recall: {test_recall:.2f}%')

            plt.legend()
            plt.grid(True)

            os.makedirs("plotsGeneralizationLoss", exist_ok=True)
            filename = f"plotsGeneralizationLoss/{num_layers}_layers_{activation_name}_epoch.png"
            plt.savefig(filename)
            print(f"Ho salvato il file {filename}")
            plt.close()
        else:
            print("No data available to create the training graph.")


