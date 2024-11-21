import tensorflow as tf
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

DATA_PATH = "C:\\Users\\GOOD\\Desktop\\Komil\\data_LibriSeVoc1.json"

def load_data(data_path):
    """Loads training dataset from a JSON file."""
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    print(f"Loaded dataset: {X.shape} MFCCs, {y.shape} labels")
    return X, y


def prepare_datasets(test_size, validation_size):
    """Splits data into training, validation, and test sets."""
    X, y = load_data(DATA_PATH)

    # Create train, validation, and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # Add a channel dimension (required for CNN input)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Generates a CNN model with regularization."""
    model = tf.keras.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=input_shape,
        kernel_regularizer=regularizers.l2(0.001)  # L2 regularization
    ))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    ))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(
        32, (2, 2), activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # Flatten output and feed into dense layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.5))  # Dropout for regularization

    # Output layer
    model.add(tf.keras.layers.Dense(8, activation='softmax'))

    return model


def plot_history(history):
    """Plots accuracy/loss for training/validation sets."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2)

    # Accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # Error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":
    # Prepare datasets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # Compile model
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # Train model
    history = model.fit(X_train, y_train,
                        validation_data=(X_validation, y_validation),
                        batch_size=32,
                        epochs=50,
                        callbacks=[early_stopping])

    # Save the model
    MODEL_SAVE_PATH = "trained_model_no_gpu.h5"
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Plot accuracy/error
    plot_history(history)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc}")

# 4573/4573 ━━━━━━━━━━━━━━━━━━━━ 16s 4ms/step - accuracy: 0.9235 - loss: 0.2696 - val_accuracy: 0.9224 - val_loss: 0.2671
# Epoch 38/50
# 4573/4573 ━━━━━━━━━━━━━━━━━━━━ 16s 4ms/step - accuracy: 0.9247 - loss: 0.2667 - val_accuracy: 0.9007 - val_loss: 0.3284
# Epoch 39/50
# 4573/4573 ━━━━━━━━━━━━━━━━━━━━ 16s 4ms/step - accuracy: 0.9235 - loss: 0.2669 - val_accuracy: 0.9138 - val_loss: 0.2994
# Epoch 40/50
# 4573/4573 ━━━━━━━━━━━━━━━━━━━━ 16s 4ms/step - accuracy: 0.9250 - loss: 0.2632 - val_accuracy: 0.9269 - val_loss: 0.2519
# Epoch 41/50
# 4573/4573 ━━━━━━━━━━━━━━━━━━━━ 16s 4ms/step - accuracy: 0.9246 - loss: 0.2656 - val_accuracy: 0.9263 - val_loss: 0.2514
# WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
# Model saved to trained_model_no_gpu.h5
# 1906/1906 - 3s - 2ms/step - accuracy: 0.9304 - loss: 0.2445

# Test accuracy: 0.9304282069206238