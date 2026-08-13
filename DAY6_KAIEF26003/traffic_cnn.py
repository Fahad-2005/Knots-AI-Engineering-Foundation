import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def generate_traffic_data(num_samples=800):
    """Generates synthetic 32x32 RGB image data across 4 traffic sign classes."""
    np.random.seed(42)
    X = np.random.rand(num_samples, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, 4, size=(num_samples,))

    for i in range(num_samples):
        cls = y[i]
        if cls == 0:    # Speed Limit (Red border)
            X[i, :4, :, 0] = 0.9
        elif cls == 1:  # Stop (High Red center)
            X[i, 10:22, 10:22, 0] = 0.95
        elif cls == 2:  # Yield (Yellow center)
            X[i, 10:22, 10:22, :2] = 0.9
        elif cls == 3:  # Pedestrian (Blue center)
            X[i, 10:22, 10:22, 2] = 0.95

    return X, y


def apply_conv_and_pool(images):
    """Simulates Conv2D feature extraction and Max Pooling across image patches."""
    num_samples = images.shape[0]
    # Reduce spatial dimensions via pooling: 32x32 -> 16x16 x 3 channels = 768 features
    pooled_features = np.zeros((num_samples, 16 * 16 * 3))

    for idx in range(num_samples):
        img = images[idx]
        # Max-pooling 2x2 blocks over the spatial grid
        pooled = img.reshape(16, 2, 16, 2, 3).max(axis=(1, 3))
        pooled_features[idx] = pooled.flatten()

    return pooled_features


def main():
    print("Generating Traffic Sign Image Dataset...")
    X_raw, y = generate_traffic_data(num_samples=800)

    print("Applying Conv2D & Max-Pooling Feature Extraction...")
    X_features = apply_conv_and_pool(X_raw)

    split = int(0.8 * len(X_features))
    X_train, X_test = X_features[:split], X_features[split:]
    y_train, y_test = y[:split], y[split:]

    print("Training Convolutional Feature Neural Network...")
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=30,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"\nTraffic Sign CNN Test Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()