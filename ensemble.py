import numpy as np
from item_response import irt, sigmoid
from utils import load_train_csv, load_valid_csv, load_public_test_csv
import matplotlib.pyplot as plt
import random


def bootstrap_data(data):
    """Generate a bootstrapped dataset by sampling with replacement."""
    n = len(data["user_id"])
    indices = np.random.choice(n, n, replace=True)
    return {
        "user_id": [data["user_id"][i] for i in indices],
        "question_id": [data["question_id"][i] for i in indices],
        "is_correct": [data["is_correct"][i] for i in indices],
    }


def ensemble_predict(models, data):
    """Compute ensemble predictions by averaging over base model outputs."""
    predictions = np.zeros(len(data["user_id"]))

    for theta, beta in models:
        for idx in range(len(data["user_id"])):
            u = data["user_id"][idx]
            q = data["question_id"][idx]
            predictions[idx] += sigmoid(theta[u] - beta[q])

    predictions /= len(models)
    return predictions


def compute_accuracy(data, predictions, threshold=0.5):
    """Evaluate accuracy given true labels and predicted probabilities."""
    return np.mean((predictions >= threshold) == np.array(data["is_correct"]))


def plot_ensemble_accuracies(val_accs, test_accs):
    plt.figure()
    model_count = list(range(1, len(val_accs) + 1))
    plt.plot(model_count, val_accs, label="Validation Accuracy", marker="o")
    plt.plot(model_count, test_accs, label="Test Accuracy", marker="x")
    plt.xlabel("Number of Base Models in Ensemble")
    plt.ylabel("Accuracy")
    plt.title("Ensemble Accuracy vs Number of Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ensemble_irt_accuracy.png")
    plt.show()


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    lr = 0.005
    iterations = 25
    num_models = 3

    models = []
    val_accs = []
    test_accs = []

    for i in range(num_models):
        print(f"Training IRT model #{i+1}")
        boot_data = bootstrap_data(train_data)
        theta, beta, *_ = irt(boot_data, val_data, lr, iterations)
        models.append((theta, beta))

        # Evaluate current ensemble of i+1 models
        val_preds = ensemble_predict(models, val_data)
        test_preds = ensemble_predict(models, test_data)

        val_acc = compute_accuracy(val_data, val_preds)
        test_acc = compute_accuracy(test_data, test_preds)

        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(f"Ensemble size: {i+1}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}\n")

    plot_ensemble_accuracies(val_accs, test_accs)


if __name__ == "__main__":
    main()
