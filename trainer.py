import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from hyper_params import HyperParams, NetworkHyperParams
from logger import log_results
from network import MNISTClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

best_iteration_case = -1
best_val_loss = sys.maxsize

class Trainer:
    def __init__(self,
                 model: MNISTClassifier,
                 hyper_params: HyperParams,
                 training_ds: DataLoader,
                 validation_ds: DataLoader,
                 testing_ds: DataLoader,
                 device: str,
                 network_hyper_params: NetworkHyperParams,
                 iteration: int
                 ):
        self.model = model
        self.hyper_params = hyper_params
        self.training_ds = training_ds
        self.validation_ds = validation_ds
        self.testing_ds = testing_ds
        self.device = device
        self.network_hyper_params = network_hyper_params
        self.iteration = iteration

        # Initialize lists to store training and validation losses
        self.train_losses = []
        self.validation_losses = []

    def train(self):
        global best_val_loss, best_iteration_case
        mnist_classifier = self.model.to(self.device)
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
                train_loss = params.error_function(train_output.to(self.device), label.to(self.device))

                train_loss.backward()
                params.optimizer.step()

                total_train_loss += train_loss.item()
                num_train_batches += 1

            avg_train_loss = total_train_loss / num_train_batches
            self.train_losses.append(avg_train_loss)  # Save training loss

            total_validation_loss = 0.0
            num_val_batches = 0
            correct_predictions = 0
            total_samples = 0

            mnist_classifier.eval()
            for val_batch, val_label in self.validation_ds:
                output_validation = mnist_classifier.forward(val_batch.to(self.device))
                validation_loss = params.error_function(output_validation.to(self.device), val_label.to(self.device))
                total_validation_loss += validation_loss.item()

                _, predicted = output_validation.max(1)
                correct_predictions += (predicted == val_label.to(self.device)).sum().item()
                total_samples += val_label.size(0)
                num_val_batches += 1

            avg_validation_loss = total_validation_loss / num_val_batches
            self.validation_losses.append(avg_validation_loss)  # Save validation loss
            validation_accuracy = 100 * correct_predictions / total_samples

            print(f"Epoch {e + 1}/{params.epochs}, Average Training Loss: {avg_train_loss}, "
                  f"Average Validation Loss: {avg_validation_loss}")
            print(f"Val accuracy ({validation_accuracy:.2f}%)")

            if num_of_validation_try > params.patience or e == params.epochs - 1:
                log_results(self.iteration, {"type": "train", "number_of_epochs" : e, "val_loss": prev_val_loss, "val_acc": validation_accuracy})
                if prev_val_loss < best_val_loss:
                    best_val_loss = prev_val_loss
                    best_iteration_case = self.iteration
                    print(f"Current best iteration: {best_iteration_case}")
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

        all_labels = []
        all_predictions = []

        for images, labels in self.testing_ds:
            outputs = self.model(images.to(self.device))
            predicted_class_indices = torch.argmax(outputs, dim=1)

            # Aggiungi le etichette e le predizioni per la matrice di confusione
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_class_indices.cpu().numpy())

            good_test += (predicted_class_indices == labels.to(self.device)).sum().item()
            total_test += labels.size(0)

        cm = confusion_matrix(all_labels, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f'./confusion_matrix/{self.iteration}_confusion_matrix.png', format='png', dpi=300)
        plt.close()

        accuracy = (good_test / total_test) * 100
        print(f"Test Accuracy: {good_test}/{total_test} ({accuracy:.2f}%)")
        log_results(self.iteration, {"type": "test", "test_accuracy" : accuracy})

    def plot_training_graph(self):
        min_length = min(len(self.train_losses), len(self.validation_losses), self.hyper_params.epochs)
        train_losses = self.train_losses[:min_length]
        validation_losses = self.validation_losses[:min_length]
        epochs = range(1, min_length + 1)

        best_epoch = self.validation_losses.index(min(self.validation_losses)) + 1
        best_loss = self.validation_losses[best_epoch - 1]

        print("Making plot...")

        plt.figure(figsize=(10, 6))

        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, validation_losses, label='Validation Loss')

        plt.scatter(best_epoch, best_loss, color='red', label=f"Early Stopping (Epoch {best_epoch})", zorder=5)

        plt.annotate(f"Min Val Loss: {best_loss:.4f}",
                     (best_epoch, best_loss),
                     textcoords="offset points",
                     xytext=(-20, 10), ha='center', color='red')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        activation_functions = ', '.join([x.value[0] for x in self.network_hyper_params.activation_fun])
        plt.title(
            f'Training and Validation Losses\nList of neurons for each hidden layer: {self.network_hyper_params.hidden_layer} \nActivation function: {activation_functions}')

        plt.legend()
        plt.grid(True)

        plt.savefig(f"./functions_plots/{self.iteration}_function.png", format='png', dpi=300)
        plt.show()
        plt.close()

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


    # def plot_testing_graph(self, good_test, total_test):
    #     accuracy = (good_test / total_test) * 100
    #     # Plot testing accuracy
    #     plt.figure(figsize=(6, 4))
    #     plt.bar(['Test Accuracy'], [accuracy])
    #     plt.ylabel('Accuracy (%)')
    #     plt.title('Testing Accuracy')
    #     plt.ylim(0, 100)
    #     plt.show()



