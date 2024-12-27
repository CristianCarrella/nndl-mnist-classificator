import sys

import torch
from torch.utils.data import Subset, DataLoader

from hyper_params import HyperParams
from network import MNISTClassifier

class Trainer:

    def __init__(self, model: MNISTClassifier, hyper_params: HyperParams,
                 training_ds: DataLoader, validation_ds: DataLoader, testing_ds: DataLoader,
                 device: str):
        self.model = model
        self.hyper_params = hyper_params
        self.training_ds = training_ds
        self.validation_ds = validation_ds
        self.testing_ds = testing_ds
        self.device = device

    def batch_train(self):
        mnist_classifier = self.model
        params = self.hyper_params
        prev_val_loss = sys.maxsize
        num_of_validation_try = 0

        for e in range(self.hyper_params.epochs):
            mnist_classifier.train()

            for batch, label in self.training_ds:
                params.optimizer.zero_grad()

                train_output = mnist_classifier.forward(batch.to(self.device))
                train_loss = params.error_function(train_output, label.to(self.device))

                train_loss.backward()
                params.optimizer.step()

                # print(f"Epoch {e + 1}/{params.epochs}, Batch Loss: {train_loss}")

            total_validation_loss = 0.0
            num_val_batches = 0

            for val_batch, val_label in self.validation_ds:
                output_validation = mnist_classifier.forward(val_batch.to(self.device))
                validation_loss = params.error_function(output_validation, val_label.to(self.device))
                total_validation_loss += validation_loss
                num_val_batches += 1

            avg_validation_loss = total_validation_loss / num_val_batches
            print(f"Epoch {e + 1}/{params.epochs}, Average Validation Loss: {avg_validation_loss}")

            if num_of_validation_try > 5:
                break
            else:
                if avg_validation_loss < prev_val_loss:
                    prev_val_loss = avg_validation_loss
                    mnist_classifier.save_model()
                else:
                    num_of_validation_try += 1

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

            for predicted, label in zip(predicted_class_indices, labels):
                if predicted == label:
                    print(f"Predicted = {predicted}, Label = {label}")

        print(f"Accuracy: {good_test}/{total_test} ({(good_test / total_test) * 100:.2f}%)")
