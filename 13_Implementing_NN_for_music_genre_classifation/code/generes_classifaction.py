import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.utils import to_categorical

JSON_PATH = "C:\\Users\\GOOD\\Desktop\\Komil\\audio_preprocessing\\13_Implementing_NN_for_music_genre_classifation\\code\\data_LibriSeVoc.json"

def load_data(json_path):
    """Load data from JSON file."""
    with open(json_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print(f"Data loaded successfully with {X.shape[0]} samples")
    print(f"MFCC shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    return X, y

def preprocess_data(X, y):
    """Preprocess data: one-hot encode labels and split into train/test sets."""
    # Ensure labels are integers starting from 0 to (num_classes - 1)
    num_classes = len(set(y))
    print(f"Number of classes: {num_classes}")

    # One-hot encode labels
    y = to_categorical(y, num_classes)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, num_classes

def build_model(input_shape, num_classes):
    """Builds and compiles a neural network model."""
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == "__main__":
    # Load data
    X, y = load_data(JSON_PATH)

    # Preprocess data
    X_train, X_test, y_train, y_test, num_classes = preprocess_data(X, y)

    # Build model
    input_shape = (X.shape[1], X.shape[2])
    model = build_model(input_shape, num_classes)

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")


# Epoch 30/30
# 5336/5336 ━━━━━━━━━━━━━━━━━━━━ 10s 2ms/step - accuracy: 0.6716 - loss: 0.8558 - val_accuracy: 0.6276 - val_loss: 1.0072
# 2287/2287 ━━━━━━━━━━━━━━━━━━━━ 2s 671us/step - accuracy: 0.6291 - loss: 0.9944
# Test accuracy: 0.6276