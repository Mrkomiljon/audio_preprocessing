import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Paths to your JSON data and model file
JSON_PATH = "C:\\Users\\GOOD\\Desktop\\Komil\\audio_preprocessing\\data_LibriSeVoc1.json"
MODEL_PATH = "audio_classification_model1.pth"

def load_data(json_path):
    """Load data from JSON file."""
    with open(json_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["mfcc"], dtype=np.float32)
    y = np.array(data["labels"], dtype=np.int64)
    print(y)
    

    # Normalizing MFCC features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    print(f"Data loaded successfully with {X.shape[0]} samples")
    return X, y 

class AudioClassificationModel(nn.Module):
    def __init__(self, num_classes=7):  # Adjusted for 8 classes
        super(AudioClassificationModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(130 * 13, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x

def plot_history(history):
    """Plot accuracy and loss for training/validation set as a function of the epochs."""
    fig, axs = plt.subplots(2, figsize=(10, 6))

    axs[0].plot(history['train_accuracy'], label="Train Accuracy")
    axs[0].plot(history['val_accuracy'], label="Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluation")

    axs[1].plot(history['train_loss'], label="Train Loss")
    axs[1].plot(history['val_loss'], label="Validation Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss Evaluation")

    plt.tight_layout()
    plt.show()

def train_and_evaluate(model, train_loader, val_loader, num_epochs=100, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_correct, train_loss_total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Adjust labels to start from 0 if needed
            labels = labels - 1

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / len(train_loader.dataset)
        train_loss_avg = train_loss_total / len(train_loader)
        history["train_accuracy"].append(train_accuracy)
        history["train_loss"].append(train_loss_avg)

        # Validation phase
        model.eval()
        val_correct, val_loss_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels - 1  # Adjust labels during validation as well
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / len(val_loader.dataset)
        val_loss_avg = val_loss_total / len(val_loader)
        history["val_accuracy"].append(val_accuracy)
        history["val_loss"].append(val_loss_avg)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")

        scheduler.step(val_loss_avg)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved at epoch {epoch+1} with validation accuracy: {val_accuracy:.2f}%")
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return history

if __name__ == "__main__":
    X, y = load_data(JSON_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)
    y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)

    num_workers = 8
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassificationModel().to(device)

    history = train_and_evaluate(model, train_loader, val_loader, num_epochs=100)
    plot_history(history)
