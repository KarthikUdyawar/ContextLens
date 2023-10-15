import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from src.utils.custom_BERT_classifier import CustomBERTClassifier
from src.utils.data_frame_batch_loader import DataFrameBatchLoader
from src.utils.text_dataset import TextDataset

DATA_VER = "0.1v"
MODEL_VER = "0.1v"

BREACH_NUM = 0

model_version_folder = f"src/model/{MODEL_VER}/checkpoint_{BREACH_NUM}"
last_model_version_folder = f"src/model/{MODEL_VER}/checkpoint_{BREACH_NUM-1}"

if not os.path.exists(model_version_folder):
    os.makedirs(model_version_folder)

train_file_path = f"src/data/{DATA_VER}/train_data.parquet"
valid_file_path = f"src/data/{DATA_VER}/valid_data.parquet"
test_file_path = f"src/data/{DATA_VER}/test_data.parquet"

print("Load dataset")
train_df = pd.read_parquet(train_file_path, engine="pyarrow")
valid_df = pd.read_parquet(valid_file_path, engine="pyarrow")
test_df = pd.read_parquet(test_file_path, engine="pyarrow")
print("Done Load dataset\n")

batch_size = 6 * 5000  # Adjust the batch size according to your needs
stratify_col = "target"

train_df_batches = DataFrameBatchLoader(
    train_df, batch_size, stratify=True, stratify_col=stratify_col
)
valid_df_batches = DataFrameBatchLoader(valid_df, batch_size)
test_df_batches = DataFrameBatchLoader(test_df, batch_size)

try:
    train_df = train_df_batches[BREACH_NUM]
    valid_df = valid_df_batches[BREACH_NUM]
    test_df = test_df_batches[BREACH_NUM]
except IndexError as e:
    print(e)


print(f"Batch: {BREACH_NUM}/{len(train_df_batches)}")

print(train_df.target.value_counts())
print(valid_df.target.value_counts())
print(test_df.target.value_counts())

print("Load dataloader")

# One hot encoding
one_hot = OneHotEncoder()
train_data = one_hot.fit_transform(train_df[["target"]]).toarray().tolist()
val_data = one_hot.fit_transform(valid_df[["target"]]).toarray().tolist()
test_data = one_hot.fit_transform(test_df[["target"]]).toarray().tolist()

train_texts = train_df["text"].tolist()
val_texts = valid_df["text"].tolist()
test_texts = test_df["text"].tolist()

max_length = 100
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = TextDataset(train_texts, train_data)
val_dataset = TextDataset(val_texts, val_data)
test_dataset = TextDataset(test_texts, test_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)
print("Done Load dataloader")

model = CustomBERTClassifier(num_classes=3)
# Checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("=== GPU not found ===")
print(f"{device = }")

# Model, optimizer, criterion initialization
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = BCEWithLogitsLoss()
num_epochs = 50

model.to(device)
file_name = f"{last_model_version_folder}/model_checkpoint+{BREACH_NUM-1}.pth"

if os.path.exists(file_name):
    # File exists, so load the model checkpoint
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Loaded model from {file_name}")
else:
    print(f"File {file_name} does not exist. Model not loaded.")


def get_accuracy(predictions, real_values):
    predictions = torch.cat(predictions).cpu()
    real_values = torch.cat(real_values).cpu()

    predictions = torch.argmax(predictions, axis=1).numpy()
    real_values = torch.argmax(real_values, axis=1).numpy()

    return accuracy_score(predictions, real_values)


def train_model(model, train_dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    predictions = []
    real_values = []

    # Initialize tqdm to show progress bar
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)

    for batch_idx, batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions.append(outputs)
        real_values.append(labels)

        # Update progress bar
        loop.set_description("Train")
        loop.set_postfix(loss=loss.item())

    accuracy = get_accuracy(predictions, real_values)

    return total_loss / len(train_dataloader), accuracy


def test(model, test_dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    real_values = []

    # Initialize tqdm to show progress bar
    loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=True)

    for batch_idx, batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions.append(outputs)
            real_values.append(labels)

            # Update progress bar
            loop.set_description("Test")
            loop.set_postfix(loss=loss.item())

    accuracy = get_accuracy(predictions, real_values)

    return total_loss / len(test_dataloader), accuracy


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

consecutive_lower_count = 0
best_val_loss = float("inf")  # Initialize with a high value
best_model_weights = None
best_epoch = None

print("Train Model")

# Training loop
for epoch in range(num_epochs):
    print(f"\nEpoch: {epoch+1}/{num_epochs}")

    train_loss, train_accuracy = train_model(
        model, train_loader, optimizer, criterion, device
    )
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print(f"Train Loss: {train_loss:.4f}\t\tTrain Accuracy: {train_accuracy:.4f}")

    val_loss, val_accuracy = test(model, val_loader, criterion, device)

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Val Loss: {val_loss:.4f}\t\tVal Accuracy: {val_accuracy:.4f}")

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        consecutive_lower_count = 0
        # Save the current best model weights
        best_model_weights = model.state_dict()
        best_epoch = epoch
    else:
        consecutive_lower_count += 1
        print(f"Strick: {consecutive_lower_count}")

    if consecutive_lower_count >= 3:
        print(
            "Early stopping: Validation loss hasn't improved for 3 consecutive epochs."
        )
        break

# Load the best model weights back into the model
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

print("Done train Model")
print("Test Model")

test_loss, test_accuracy = test(model, test_loader, criterion, device)

if test_accuracy > 0.8:
    file_name = f"{model_version_folder}/model_checkpoint+{BREACH_NUM}.pth"
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": test_loss,
        },
        file_name,
    )
    print("Model saved")
    print(f"{test_loss = }, {test_accuracy = }")
else:
    print(f"Bad Model {test_loss = }, {test_accuracy = }")

print("Done test Model")

print("Getting reports")


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    # Create a DataFrame to hold the training history data
    history_data = pd.DataFrame(
        {
            "Epoch": range(1, len(train_losses) + 1),
            "Train Loss": train_losses,
            "Validation Loss": val_losses,
            "Train Accuracy": train_accuracies,
            "Validation Accuracy": val_accuracies,
        }
    )

    # Save the data to a CSV file
    csv_file_name = f"{model_version_folder}/training_history+{BREACH_NUM}.csv"
    history_data.to_csv(csv_file_name, index=False)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(1, len(train_losses) + 1), y=train_losses, label="Train Loss")
    sns.lineplot(x=range(1, len(val_losses) + 1), y=val_losses, label="Validation Loss")
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
    plt.savefig(f"{model_version_folder}/Training&Validation-Losses+{BREACH_NUM}.png")
    plt.show()


plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)


# Finding Accuracy
def get_predictions(model, data_loader, predict_proba=False):
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


y_pred, y_test = get_predictions(model, test_loader)


def save_classification_report(true_labels, predicted_labels, target_names, file_path):
    # Generate the classification report
    report = classification_report(
        true_labels, predicted_labels, target_names=target_names
    )

    # Save the classification report to a text file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(report)


report_file_path = f"{model_version_folder}/report+{BREACH_NUM}.txt"

save_classification_report(
    y_test,
    y_pred,
    one_hot.categories_[0].tolist(),
    report_file_path,
)


def save_confusion_matrix(BREACH_NUM, model_version_folder, one_hot, y_pred, y_test):
    conf_mat = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(
        conf_mat, display_labels=one_hot.categories_[0].tolist()
    ).plot()
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.savefig(f"{model_version_folder}/ConfusionMatrixDisplay+{BREACH_NUM}.png")


save_confusion_matrix(BREACH_NUM, model_version_folder, one_hot, y_pred, y_test)

print("Done getting reports")

print("Complete")
