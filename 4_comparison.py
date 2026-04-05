import json
import os


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    traditional_results_path = os.path.join(base_dir, "traditional_ml_results.json")
    cnn_results_path = os.path.join(base_dir, "cnn_results.json")

    if not os.path.exists(traditional_results_path):
        raise FileNotFoundError(
            f"Missing {traditional_results_path}. Run 2_traditional_ml.py first."
        )

    if not os.path.exists(cnn_results_path):
        raise FileNotFoundError(
            f"Missing {cnn_results_path}. Run 3_cnn_model.py first."
        )

    traditional_results = load_json_file(traditional_results_path)
    cnn_results = load_json_file(cnn_results_path)

    rows = [
        ("Decision Tree", traditional_results["decision_tree_accuracy"] * 100.0),
        ("SVM", traditional_results["svm_accuracy"] * 100.0),
        ("K-Means (purity)", traditional_results["kmeans_purity"] * 100.0),
        ("CNN", cnn_results["cnn_accuracy"] * 100.0),
    ]

    print("Model              | Test Accuracy")
    print("-------------------|---------------")
    for model_name, accuracy_value in rows:
        print(f"{model_name:<19}| {accuracy_value:>6.2f}%")


if __name__ == "__main__":
    main()
