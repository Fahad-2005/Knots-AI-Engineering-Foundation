import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_mnist_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = build_mnist_model()
    print("\nModel Summary:")
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    print("\nTraining MNIST Classifier...")
    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        callbacks=callbacks
    )

    print("\nEvaluating on Test Data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()