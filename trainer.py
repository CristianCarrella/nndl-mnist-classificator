import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from hyper_params import HyperParams, NetworkHyperParams
from network import MNISTClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class NetworkResults:
    def __init__(self):
        self.iteration = 0
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        self.confusion_matrix_result = []

    def get_iteration(self):
        self.iteration +=1
        return self.iteration - 1


class Trainer:
    def __init__(self,
                 model: MNISTClassifier,
                 hyper_params: HyperParams,
                 training_ds: DataLoader,
                 validation_ds: DataLoader,
                 testing_ds: DataLoader,
                 device: str,
                 network_hyper_params: NetworkHyperParams,
                 network_results = NetworkResults()
                 ):
        self.model = model
        self.hyper_params = hyper_params
        self.training_ds = training_ds
        self.validation_ds = validation_ds
        self.testing_ds = testing_ds
        self.device = device
        self.network_hyper_params = network_hyper_params
        self.network_results = network_results

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
        plt.savefig(f'./confusion_matrix/{self.network_results.get_iteration()}_confusion_matrix.png', format='png', dpi=300)
        plt.close()

        accuracy = (good_test / total_test) * 100
        print(f"Test Accuracy: {good_test}/{total_test} ({accuracy:.2f}%)")

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


    def log_results(self):
        f = open("log.txt", "a")
        f.write()
        f.close()
