import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("Loading Handwritten Digits Dataset...")
    # Load 8x8 handwritten digits directly from scikit-learn (no network/pandas required)
    digits = load_digits()
    X, y = digits.data, digits.target

    # Normalize pixel values from [0, 16] to [0.0, 1.0]
    X = X / 16.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("\nTraining Neural Network (MLPClassifier)...")
    # Multi-Layer Perceptron: Input(64) -> Hidden1(64) -> Hidden2(32) -> Output(10)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=100,
        early_stopping=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\nEvaluating Model...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()