import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

DATA_PATH = "C:\\Users\\GOOD\\Desktop\\Komil\\filtered_dataset.csv"

def load_data(data_path):
    """Loads training dataset from CSV file."""
    data = pd.read_csv(data_path)

    # Xususiyatlar va nishonlarni ajratish
    X = data[['zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'rmse', 'mfcc_13']].values
    y = data['label'].values

    # Ma'lumotlarni standartlashtirish
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3D formatga o'tkazish (LSTM uchun kerak)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set."""
    fig, axs = plt.subplots(2)

    # Accuracy grafigi
    axs[0].plot(history.history["accuracy"], label="Train Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Eval")

    # Loss grafigi
    axs[1].plot(history.history["loss"], label="Train Loss")
    axs[1].plot(history.history["val_loss"], label="Validation Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss Eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Splits data into train, validation, and test sets."""
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, num_classes):
    """Generates RNN-LSTM model with Bidirectional LSTM and dropout."""
    model = keras.Sequential()

    # Bidirectional LSTM layers
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128)))

    # Dense and Dropout layers
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.4))  # Dropout to prevent overfitting

    # Output layer
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    return model


if __name__ == "__main__":

    # Split data
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # Create network
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    model = build_model(input_shape, num_classes)

    # Compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Class weights for balanced dataset
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    # Define callbacks
    checkpoint = ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=1,
        restore_best_weights=True
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        batch_size=32,
        epochs=50,
        callbacks=[checkpoint, early_stopping],
        class_weight=class_weights  # Adding class weights
    )

    # Plot accuracy/loss for training and validation
    plot_history(history)

    # Evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    print("Best model saved as 'best_model.keras'")


# 1732/1733 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.4392 - loss: 1.4057  
# Epoch 23: val_loss improved from 1.44525 to 1.44067, saving model to best_model.keras
# 1733/1733 ━━━━━━━━━━━━━━━━━━━━ 8s 4ms/step - accuracy: 0.4392 - loss: 1.4057 - val_accuracy: 0.4231 - val_loss: 1.4407
# Epoch 24/50
# 1726/1733 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.4423 - loss: 1.3984  
# Epoch 24: val_loss did not improve from 1.44067
# 1733/1733 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.4423 - loss: 1.3984 - val_accuracy: 0.4245 - val_loss: 1.4412
# Epoch 25/50
# 1725/1733 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.4458 - loss: 1.3882  
# Epoch 25: val_loss improved from 1.44067 to 1.43741, saving model to best_model.keras
# 1733/1733 ━━━━━━━━━━━━━━━━━━━━ 10s 6ms/step - accuracy: 0.4458 - loss: 1.3883 - val_accuracy: 0.4304 - val_loss: 1.4374
# Epoch 26/50
# 1731/1733 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.4490 - loss: 1.3893  
# Epoch 26: val_loss did not improve from 1.43741
# 1733/1733 ━━━━━━━━━━━━━━━━━━━━ 12s 7ms/step - accuracy: 0.4490 - loss: 1.3893 - val_accuracy: 0.4241 - val_loss: 1.4465
# Epoch 27/50
# 1726/1733 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.4518 - loss: 1.3748  
# Epoch 27: val_loss did not improve from 1.43741
# 1733/1733 ━━━━━━━━━━━━━━━━━━━━ 12s 7ms/step - accuracy: 0.4518 - loss: 1.3749 - val_accuracy: 0.4238 - val_loss: 1.4469
# Epoch 28/50
# 1729/1733 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.4550 - loss: 1.3705  
# Epoch 28: val_loss did not improve from 1.43741
# 1733/1733 ━━━━━━━━━━━━━━━━━━━━ 12s 7ms/step - accuracy: 0.4550 - loss: 1.3705 - val_accuracy: 0.4272 - val_loss: 1.4487
# Epoch 29/50
# 1727/1733 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.4559 - loss: 1.3628  
# Epoch 29: val_loss did not improve from 1.43741
# 1733/1733 ━━━━━━━━━━━━━━━━━━━━ 12s 7ms/step - accuracy: 0.4559 - loss: 1.3628 - val_accuracy: 0.4257 - val_loss: 1.4484
# Epoch 30/50
# 1726/1733 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - accuracy: 0.4601 - loss: 1.3505  
# Epoch 30: val_loss did not improve from 1.43741
# 1733/1733 ━━━━━━━━━━━━━━━━━━━━ 12s 7ms/step - accuracy: 0.4600 - loss: 1.3505 - val_accuracy: 0.4259 - val_loss: 1.4441
# Epoch 30: early stopping
# Restoring model weights from the end of the best epoch: 25.
# 722/722 - 2s - 3ms/step - accuracy: 0.4286 - loss: 1.4471