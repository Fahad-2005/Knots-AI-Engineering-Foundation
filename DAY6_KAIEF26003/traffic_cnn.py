import numpy as np
import tensorflow as tf
from tensorflow import keras


def generate_synthetic_traffic_data(num_samples=1200):
    """Generates synthetic 32x32 RGB images representing 4 traffic sign categories."""
    np.random.seed(42)
    # Generate random 32x32x3 images
    X = np.random.rand(num_samples, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, 4, size=(num_samples,))

    # Embed distinctive visual features per category
    for i in range(num_samples):
        cls = y[i]
        if cls == 0:    # Speed Limit (Red border)
            X[i, :4, :, 0] = 0.9
        elif cls == 1:  # Stop (High Red intensity center)
            X[i, 10:22, 10:22, 0] = 0.95
        elif cls == 2:  # Yield (High Yellow / Red-Green combo)
            X[i, 10:22, 10:22, :2] = 0.9
        elif cls == 3:  # Pedestrian (High Blue intensity)
            X[i, 10:22, 10:22, 2] = 0.95

    return X, y


def build_traffic_cnn(input_shape=(32, 32, 3), num_classes=4):
    model = keras.Sequential([
        # Convolutional Block 1
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),

        # Convolutional Block 2
        keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2)),

        # Dense Classifier Head
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print("Generating Traffic Sign Image Dataset...")
    X, y = generate_synthetic_traffic_data(num_samples=1200)

    # Train/Test Split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = build_traffic_cnn()
    print("\nTraffic Sign CNN Architecture:")
    model.summary()

    print("\nTraining CNN Model...")
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.15
    )

    print("\nEvaluating CNN Performance...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()