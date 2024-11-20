import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import gc
import csv

# CSV fayl va model faylining yo'li
CSV_PATH = r"C:\\Users\\GOOD\\Desktop\\audio_features_dataset.csv"
MODEL_PATH = "balanced_audio_classification_model.pth"

# CUDA qurilmasi mavjudligini tekshirish
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSV Faylni Yuklash va Preprocessing
def load_csv_in_chunks(csv_path, chunk_size=10000):
    """CSV faylni qismlarga bo'lib yuklash."""
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, delimiter=',', encoding='utf-8', on_bad_lines='skip'):
        X_chunk = chunk.drop(columns=['filename', 'label']).values
        y_chunk = chunk['label'].values
        X_chunk = torch.tensor(X_chunk, dtype=torch.float32)
        y_chunk = torch.tensor(y_chunk, dtype=torch.long)
        yield X_chunk, y_chunk

# Datasetni qismlarga bo'lib yuklash
X_all, y_all = [], []
for X_chunk, y_chunk in load_csv_in_chunks(CSV_PATH):
    X_all.append(X_chunk)
    y_all.append(y_chunk)

# To'liq ma'lumotni birlashtirish
X = torch.cat(X_all)
y = torch.cat(y_all)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Har bir sinf uchun ma'lumotlar sonini hisoblash
unique_classes, counts = np.unique(y_train, return_counts=True)
print(f"Class distribution: {dict(zip(unique_classes, counts))}")

# Class_weights hisoblash
class_weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# TensorDataset va DataLoader yaratish
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# WeightedRandomSampler yordamida balanslash
class_sample_count = np.array([len(np.where(y_train.numpy() == t)[0]) for t in np.unique(y_train.numpy())])
weights = 1. / class_sample_count
samples_weights = np.array([weights[t] for t in y_train.numpy()])
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

batch_size = 64
num_workers = 8

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, prefetch_factor=2
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2
)

# Model Arxitekturasi
class AudioClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes=7):
        super(AudioClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.output(x)

# Modelni yaratish
model = AudioClassificationModel(input_size=X_train.shape[1]).to(device)

# Trening Funksiyasi
def train_model(model, train_loader, test_loader, num_epochs=20):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Accuracy: {train_accuracy:.2f}%")
        
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / len(test_loader.dataset)
        print(f"Validation Accuracy: {test_accuracy:.2f}%")
        scheduler.step(train_loss)

        # GPU va CPU xotirasini tozalash
        torch.cuda.empty_cache()
        gc.collect()

    return model

# Modelni o'qitish
trained_model = train_model(model, train_loader, test_loader)

# Modelni saqlash
torch.save(trained_model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
