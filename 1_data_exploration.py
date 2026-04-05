import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10


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


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    train_distribution = np.bincount(y_train, minlength=len(CLASS_NAMES))
    test_distribution = np.bincount(y_test, minlength=len(CLASS_NAMES))

    print("Training data shape:", x_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test data shape:", x_test.shape)
    print("Test labels shape:", y_test.shape)
    print("\nTraining class distribution:")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index} - {class_name}: {train_distribution[index]}")

    print("\nTest class distribution:")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index} - {class_name}: {test_distribution[index]}")

    plt.figure(figsize=(14, 6))
    for plot_index in range(10):
        plt.subplot(2, 5, plot_index + 1)
        plt.imshow(x_train[plot_index])
        plt.title(CLASS_NAMES[y_train[plot_index]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(CLASS_NAMES, train_distribution, color="steelblue")
    plt.title("CIFAR-10 Training Set Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
