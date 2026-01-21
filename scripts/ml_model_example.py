#!/usr/bin/env python3
"""
ML Model Example

This script demonstrates a simple machine learning workflow using scikit‑learn.
We define a class `MLModelExample` that loads the Iris dataset, splits it,
trains a logistic regression model, and evaluates accuracy.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class MLModelExample:
    def __init__(self):
        """Initialize the class with placeholders for the model and data."""
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = (None, None, None, None)

    def load_data(self):
        """
        Load the Iris dataset and split it into training and testing sets.
        """
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print("Data loaded and split into train/test sets.")

    def train_model(self):
        """
        Train a logistic regression classifier on the training data.
        """
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set and print accuracy.
        """
        if self.model is None:
            print("Model has not been trained.")
            return
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Model accuracy: {acc:.4f}")


if __name__ == "__main__":
    # Example usage
    example = MLModelExample()
    example.load_data()    # Load and split the Iris dataset
    example.train_model()   # Train the logistic regression model
    example.evaluate_model()  # Evaluate and print accuracy
