# KNOWN LIMITATIONS (matches original project findings):
# - No batch normalisation implemented (would improve accuracy)
# - Hardware constraints limit epochs to 50
# - State-of-the-art CIFAR-10 accuracy is ~99%; this model targets ~75-80%
# - SVM trained on subset (10k samples) due to memory constraints

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_model():
    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(32, 32, 3),
            ),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )
    return model


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "cifar10_cnn_model.h5")
    results_path = os.path.join(base_dir, "cnn_results.json")

    np.random.seed(42)
    tf.random.set_seed(42)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    y_train_encoded = to_categorical(y_train, num_classes=len(CLASS_NAMES))
    y_test_encoded = to_categorical(y_test, num_classes=len(CLASS_NAMES))

    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train_encoded,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)
    model.save(model_path)

    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Saved CNN model to: {model_path}")

    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump({"cnn_accuracy": float(test_accuracy)}, results_file, indent=2)

    print(f"Saved CNN results to: {results_path}")
    plot_training_history(history)


if __name__ == "__main__":
    main()
