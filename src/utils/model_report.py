"""Designed for creating reports and visualizations for model training and evaluation"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


class ModelReportManager:
    """_summary_"""
    def __init__(
        self, model_version_folder, breach_num, save_reports=True, verbose=True
    ):
        """
        Initialize the ModelReportManager.

        Args:
            model_version_folder: The folder where reports are saved.
            breach_num: An identifier for the model's performance.
            save_reports: If True, reports will be saved to files.
            verbose: If True, reports will be printed to the console.
        """
        self.model_version_folder = model_version_folder
        self.save_reports = save_reports
        self.breach_num = breach_num
        self.verbose = verbose

    def plot_training_history(
        self,
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
    ):
        """
        Plot and optionally save training history graphs.

        Args:
            train_losses: Training losses.
            val_losses: Validation losses.
            train_accuracies: Training accuracies.
            val_accuracies: Validation accuracies.
        """
        history_data = pd.DataFrame(
            {
                "Epoch": range(1, len(train_losses) + 1),
                "Train Loss": train_losses,
                "Validation Loss": val_losses,
                "Train Accuracy": train_accuracies,
                "Validation Accuracy": val_accuracies,
            }
        )

        self._plot_line_graph(
            train_losses, val_losses, train_accuracies, val_accuracies
        )

        if self.save_reports:
            csv_file_name = (
                f"{self.model_version_folder}/training_history+{self.breach_num}.csv"
            )
            history_data.to_csv(csv_file_name, index=False)
            plt.savefig(
                f"{self.model_version_folder}/Training&Validation-Losses+{self.breach_num}.png"
            )
        if self.verbose:
            self._print_verbose("Losses vs epoch", history_data)

    def get_predictions(self, model, data_loader, device, predict_proba=False):
        """
        Get predictions from the model.

        Args:
            model: The classification model.
            data_loader: DataLoader for data.
            device: Device (CPU or GPU).
            predict_proba: If True, return probability predictions.

        Returns:
            Predicted labels and true labels.
        """
        model = model.eval()
        predictions = []
        real_values = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                predictions.append(outputs)
                real_values.append(labels)

        y_pred = torch.cat(predictions).cpu()
        y_test = torch.cat(real_values).cpu()

        if not predict_proba:
            y_pred = torch.argmax(y_pred, axis=1).numpy()
            y_test = torch.argmax(y_test, axis=1).numpy()

        return y_pred, y_test

    def save_classification_report(self, true_labels, predicted_labels, target_names):
        """
        Generate and optionally save a classification report.

        Args:
            true_labels: True labels.
            predicted_labels: Predicted labels.
            target_names: Names of target classes.
        """
        report = classification_report(
            true_labels, predicted_labels, target_names=target_names
        )
        if self.save_reports:
            file_path = f"{self.model_version_folder}/report+{self.breach_num}.txt"

            with open(file_path, "w", encoding="utf-8") as file:
                file.write(report)
        if self.verbose:
            self._print_verbose("Classification report", report)

    def save_confusion_matrix(self, one_hot, y_pred, y_test):
        """
        Generate and optionally save a confusion matrix.

        Args:
            one_hot: One-hot encoding (categories).
            y_pred: Predicted labels.
            y_test: True labels.
        """
        conf_mat = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(
            conf_mat, display_labels=one_hot.categories_[0].tolist()
        ).plot()
        plt.title("Confusion Matrix")
        plt.grid(False)

        if self.save_reports:
            plt.savefig(
                f"{self.model_version_folder}/ConfusionMatrixDisplay+{self.breach_num}.png"
            )
        if self.verbose:
            self._print_verbose("Confusion Matrix", conf_mat)

    def _plot_line_graph(
        self, train_losses, val_losses, train_accuracies, val_accuracies
    ):
        """
        Plot line graphs for training history.

        Args:
            train_losses: Training losses.
            val_losses: Validation losses.
            train_accuracies: Training accuracies.
            val_accuracies: Validation accuracies.
        """
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.lineplot(
            x=range(1, len(train_losses) + 1), y=train_losses, label="Train Loss"
        )
        sns.lineplot(
            x=range(1, len(val_losses) + 1), y=val_losses, label="Validation Loss"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Losses")

        plt.subplot(1, 2, 2)
        sns.lineplot(
            x=range(1, len(train_accuracies) + 1),
            y=train_accuracies,
            label="Train Accuracy",
        )
        sns.lineplot(
            x=range(1, len(val_accuracies) + 1),
            y=val_accuracies,
            label="Validation Accuracy",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Validation Accuracies")

        plt.tight_layout()

    def _print_verbose(self, title, report):
        """
        Print information if in verbose mode.

        Args:
            title: Information title.
            report: Information to be printed.
        """
        print(title)
        print(report)
        plt.show()
