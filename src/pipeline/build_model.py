"""Build and train the model"""
import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.custom_BERT_classifier import CustomBERTClassifier
from src.utils.data_frame_batch_loader import DataFrameBatchLoader
from src.utils.model_report import ModelReportManager
from src.utils.text_dataset import TextDataset

PWD = os.getcwd()

DATA_VER = "0.1v"
MODEL_VER = "0.1v"

BREACH_NUM = 0
BATCH_SIZE = 6 * 5000
NUM_EPOCHS = 50
ACCEPTABLE_ACC = 0.85

SAVE_MODEL_REPORTS = True

model_version_folder = f"{PWD}/src/model/{MODEL_VER}/checkpoint_{BREACH_NUM}"
last_model_version_folder = f"{PWD}/src/model/{MODEL_VER}/checkpoint_{BREACH_NUM-1}"
last_model_file_name = (
    f"{last_model_version_folder}/model_checkpoint+{BREACH_NUM-1}.pth"
)
model_file_name = f"{model_version_folder}/model_checkpoint+{BREACH_NUM}.pth"

if not os.path.exists(model_version_folder):
    os.makedirs(model_version_folder)

train_file_path = f"{PWD}/src/data/{DATA_VER}/train_data.parquet"
valid_file_path = f"{PWD}/src/data/{DATA_VER}/valid_data.parquet"
test_file_path = f"{PWD}/src/data/{DATA_VER}/test_data.parquet"

print("Load dataset")
train_df = pd.read_parquet(train_file_path, engine="pyarrow")
valid_df = pd.read_parquet(valid_file_path, engine="pyarrow")
test_df = pd.read_parquet(test_file_path, engine="pyarrow")
print("Done Load dataset\n")

train_df_batches = DataFrameBatchLoader(train_df, BATCH_SIZE, stratify_col="target")
valid_df_batches = DataFrameBatchLoader(valid_df, BATCH_SIZE)
test_df_batches = DataFrameBatchLoader(test_df, BATCH_SIZE)

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

# Model, optimizer, criterion initialization
optimizer = Adam(model.parameters(), lr=1e-5)
criterion = BCEWithLogitsLoss()

model.to(device)

if os.path.exists(last_model_file_name):
    # File exists, so load the model checkpoint
    checkpoint = torch.load(last_model_file_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("Model loaded")
else:
    print("=== Model not found ===")


def get_accuracy(predictions: torch.Tensor, real_values: torch.Tensor) -> float:
    """
    Calculate the accuracy of predictions compared to the real values.

    Args:
        predictions (torch.Tensor): Predicted values.
        real_values (torch.Tensor): Real (ground truth) values.

    Returns:
        float: The accuracy score.
    """
    predictions = torch.cat(predictions).cpu()
    real_values = torch.cat(real_values).cpu()

    predictions = torch.argmax(predictions, axis=1).numpy()
    real_values = torch.argmax(real_values, axis=1).numpy()

    return accuracy_score(predictions, real_values)


def train_model(
    _model: CustomBERTClassifier,
    train_dataloader: DataLoader,
    _optimizer: Adam,
    _criterion: BCEWithLogitsLoss,
    _device: torch.device,
) -> tuple[float, float]:
    """
    Train the BERT-based classifier model.

    Args:
        _model (CustomBERTClassifier): The custom BERT model.
        train_dataloader (DataLoader): DataLoader for training data.
        _optimizer (Adam): Optimizer for model training.
        _criterion (BCEWithLogitsLoss): Loss criterion for training.
        _device (torch.device): Device for training (CPU or GPU).

    Returns:
        tuple[float, float]: A tuple of training loss and accuracy.
    """
    _model.train()
    total_loss = 0
    predictions = []
    real_values = []

    # Initialize tqdm to show progress bar
    loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)

    for _, batch in loop:
        input_ids = batch["input_ids"].to(_device)
        attention_mask = batch["attention_mask"].to(_device)
        labels = batch["labels"].to(_device)

        _optimizer.zero_grad()
        outputs = _model(input_ids, attention_mask)
        _loss = _criterion(outputs, labels)

        _loss.backward()
        _optimizer.step()

        total_loss += _loss.item()

        predictions.append(outputs)
        real_values.append(labels)

        # Update progress bar
        loop.set_description("Train")
        loop.set_postfix(loss=_loss.item())

    accuracy = get_accuracy(predictions, real_values)

    return total_loss / len(train_dataloader), accuracy


def test(
    _model: CustomBERTClassifier,
    test_dataloader: DataLoader,
    _criterion: BCEWithLogitsLoss,
    _device: torch.device,
    valid_mode: bool = True,
) -> tuple[float, float]:
    """
    Test the BERT-based classifier model.

    Args:
        _model (CustomBERTClassifier): The custom BERT model.
        test_dataloader (DataLoader): DataLoader for testing data.
        _criterion (BCEWithLogitsLoss): Loss criterion for testing.
        _device (torch.device): Device for testing (CPU or GPU).
        valid_mode (bool, optional): Set to True for validation, False for testing.
        Defaults to True.

    Returns:
        tuple[float, float]: A tuple of testing/validation loss and accuracy.
    """
    _model.eval()
    total_loss = 0
    predictions = []
    real_values = []

    # Initialize tqdm to show progress bar
    loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=True)

    for _, batch in loop:
        input_ids = batch["input_ids"].to(_device)
        attention_mask = batch["attention_mask"].to(_device)
        labels = batch["labels"].to(_device)

        with torch.no_grad():
            outputs = _model(input_ids, attention_mask)
            _loss = _criterion(outputs, labels)

            total_loss += _loss.item()

            predictions.append(outputs)
            real_values.append(labels)

            # Update progress bar
            if valid_mode:
                loop.set_description("Val")
            else:
                loop.set_description("Test")
            loop.set_postfix(loss=_loss.item())

    accuracy = get_accuracy(predictions, real_values)

    return total_loss / len(test_dataloader), accuracy


train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# pylint: disable=invalid-name
consecutive_lower_count = 0
best_val_loss = float("inf")  # Initialize with a high value
best_model_weights = None
best_epoch = None

print("Train Model")

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch: {epoch+1}/{NUM_EPOCHS}")

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

test_loss, test_accuracy = test(model, test_loader, criterion, device, valid_mode=False)

if test_accuracy > ACCEPTABLE_ACC:
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": test_loss,
        },
        model_file_name,
    )
    print("Model saved")
    print(f"{test_loss = }, {test_accuracy = }")
else:
    print(f"Bad Model {test_loss = }, {test_accuracy = }")

print("Done test Model")

print("Getting reports")

report_manager = ModelReportManager(
    model_version_folder, BREACH_NUM, save_reports=SAVE_MODEL_REPORTS
)

# Assuming you have your data and model ready
report_manager.plot_training_history(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
)
y_pred, y_test = report_manager.get_predictions(model, test_loader, device)
report_manager.save_classification_report(
    y_test, y_pred, one_hot.categories_[0].tolist()
)
report_manager.save_confusion_matrix(one_hot, y_pred, y_test)

print("Done getting reports")

print("Complete")
