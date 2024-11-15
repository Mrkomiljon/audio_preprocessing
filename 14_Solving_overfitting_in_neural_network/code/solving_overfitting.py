import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

# Path to your JSON data
JSON_PATH = "C:\\Users\\GOOD\\Desktop\\Komil\\data_LibriSeVoc.json"

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

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs."""
    fig, axs = plt.subplots(2, figsize=(10, 6))

    # Accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluation")

    # Loss subplot
    axs[1].plot(history.history["loss"], label="train loss")
    axs[1].plot(history.history["val_loss"], label="test loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss Evaluation")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data
    X, y = load_data(JSON_PATH)

    # Preprocess data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build network topology
    model = Sequential([
        # Input layer
        Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st dense layer
        Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),

        # 2nd dense layer
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),

        # 3rd dense layer
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),

        # Output layer
        Dense(7, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

    # Plot accuracy and loss
    plot_history(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

# not good
# Epoch 20/50
# 5336/5336 [==============================] - 40s 8ms/step - loss: 1.9460 - accuracy: 0.1433 - val_loss: 1.9460 - val_accuracy: 0.1433
# Epoch 21/50
# 5336/5336 [==============================] - 41s 8ms/step - loss: 1.9460 - accuracy: 0.1433 - val_loss: 1.9460 - val_accuracy: 0.1431
# Epoch 22/50
# 5336/5336 [==============================] - 41s 8ms/step - loss: 1.9460 - accuracy: 0.1440 - val_loss: 1.9460 - val_accuracy: 0.1431
# Epoch 23/50
# 5336/5336 [==============================] - 40s 8ms/step - loss: 1.9460 - accuracy: 0.1439 - val_loss: 1.9459 - val_accuracy: 0.1440
# Epoch 24/50
# 5336/5336 [==============================] - 40s 8ms/step - loss: 1.9460 - accuracy: 0.1423 - val_loss: 1.9459 - val_accuracy: 0.1431
# Epoch 25/50
# 5336/5336 [==============================] - 41s 8ms/step - loss: 1.9460 - accuracy: 0.1429 - val_loss: 1.9459 - val_accuracy: 0.1440
# Epoch 26/50