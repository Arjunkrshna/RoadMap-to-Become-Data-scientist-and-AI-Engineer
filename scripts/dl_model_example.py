#!/usr/bin/env python3
"""
Deep Learning Model Example

This script demonstrates a basic deep learning workflow using TensorFlow and Keras.
We define a simple neural network to classify the Iris dataset.
"""

import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


class DLModelExample:
    def __init__(self):
        """Initialize placeholders for the model and data."""
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)

    def load_data(self):
        """
        Load the Iris dataset, convert labels to one-hot encoded form, and split into train/test.
        """
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        # One-hot encode labels
        y_categorical = to_categorical(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_categorical, test_size=0.3, random_state=42
        )
        print("Data loaded and preprocessed.")

    def build_model(self):
        """
        Build a simple feedforward neural network using Keras.
        """
        self.model = Sequential([
            Dense(16, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dense(16, activation='relu'),
            Dense(self.y_train.shape[1], activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model built and compiled.")

    def train(self, epochs: int = 50):
        """
        Train the neural network for a specified number of epochs.
        """
        if self.model is None:
            print("Model has not been built.")
            return
        self.model.fit(self.X_train, self.y_train, epochs=epochs, verbose=0)
        print(f"Training completed for {epochs} epochs.")

    def evaluate(self):
        """
        Evaluate the model on the test data and print loss and accuracy.
        """
        if self.model is None:
            print("Model has not been trained.")
            return
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    # Example usage
    example = DLModelExample()
    example.load_data()   # Load and preprocess the data
    example.build_model()  # Build the neural network
    example.train(epochs=50)  # Train the model
    example.evaluate()  # Evaluate the model performance
