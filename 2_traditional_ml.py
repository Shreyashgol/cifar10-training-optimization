import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics.cluster import contingency_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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


def purity_score(y_true, y_pred):
    matrix = contingency_matrix(y_true, y_pred)
    return np.sum(np.max(matrix, axis=0)) / np.sum(matrix)


def build_cluster_label_map(cluster_assignments, y_true, n_clusters):
    cluster_to_label = {}
    for cluster_index in range(n_clusters):
        members = y_true[cluster_assignments == cluster_index]
        if members.size == 0:
            cluster_to_label[cluster_index] = 0
        else:
            cluster_to_label[cluster_index] = int(
                np.bincount(members, minlength=len(CLASS_NAMES)).argmax()
            )
    return cluster_to_label


def print_metrics(model_name, y_true, y_pred, accuracy_value):
    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"Accuracy: {accuracy_value:.4f}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, "traditional_ml_results.json")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    decision_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    decision_tree.fit(x_train_flat, y_train)
    dt_predictions = decision_tree.predict(x_test_flat)
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    print_metrics("Decision Tree", y_test, dt_predictions, dt_accuracy)

    rng = np.random.default_rng(42)
    svm_indices = rng.choice(x_train_flat.shape[0], size=10000, replace=False)
    x_train_svm = x_train_flat[svm_indices]
    y_train_svm = y_train[svm_indices]

    svm_model = SVC(kernel="rbf", C=1.0)
    svm_model.fit(x_train_svm, y_train_svm)
    svm_predictions = svm_model.predict(x_test_flat)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print_metrics("SVM", y_test, svm_predictions, svm_accuracy)

    kmeans_model = KMeans(n_clusters=10, n_init=10, random_state=42)
    kmeans_model.fit(x_train_flat)
    train_clusters = kmeans_model.labels_
    cluster_label_map = build_cluster_label_map(train_clusters, y_train, 10)

    test_clusters = kmeans_model.predict(x_test_flat)
    kmeans_class_predictions = np.array(
        [cluster_label_map[cluster_index] for cluster_index in test_clusters]
    )
    kmeans_accuracy = accuracy_score(y_test, kmeans_class_predictions)
    kmeans_purity = purity_score(y_test, test_clusters)
    print_metrics("K-Means", y_test, kmeans_class_predictions, kmeans_accuracy)
    print(f"K-Means purity score: {kmeans_purity:.4f}")

    results = {
        "decision_tree_accuracy": float(dt_accuracy),
        "svm_accuracy": float(svm_accuracy),
        "kmeans_accuracy": float(kmeans_accuracy),
        "kmeans_purity": float(kmeans_purity),
    }

    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)

    print(f"\nSaved traditional ML results to: {results_path}")


if __name__ == "__main__":
    main()
