#!/usr/bin/env python3
"""
All In One Example

This script demonstrates how to use multiple Python libraries (pandas, scikit-learn, TensorFlow/Keras, and Hugging Face transformers) within a single class.
The class `AllInOneExample` loads the Iris dataset into a pandas DataFrame, summarizes the data, trains a logistic regression model (scikit-learn),
trains a simple neural network (TensorFlow/Keras), and generates text using a small GPT-2 model (transformers). Each method includes comments explaining what it does.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class AllInOneExample:
    """A class that demonstrates a complete workflow using popular data science libraries."""

    def __init__(self):
        """Initialize all attributes."""
        # Load the Iris dataset into a pandas DataFrame
        iris = load_iris()
        # Save features and target names
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        # Create a DataFrame for features
        self.df = pd.DataFrame(iris.data, columns=self.feature_names)
        # Add target column
        self.df['target'] = iris.target

        # Prepare attributes for models (will be set later)
        self.log_reg_model = None
        self.nn_model = None
        # Set up tokenizer and model for text generation (download small model)
        # Using distilgpt2 which is lightweight
        self.tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        self.text_model = AutoModelForCausalLM.from_pretrained('distilgpt2')

    def summarize(self):
        """Print basic information about the dataset."""
        print("First few rows of the data:")
        print(self.df.head())
        print("\nSummary statistics:")
        print(self.df.describe())
        print("\nClass distribution (target counts):")
        print(self.df['target'].value_counts())

    def train_logistic_regression(self):
        """Train a Logistic Regression classifier and print accuracy."""
        # Split features and target
        X = self.df[self.feature_names]
        y = self.df['target']
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Create and train the model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        # Evaluate on test set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Logistic Regression accuracy: {acc:.2f}")
        # Save model for later use
        self.log_reg_model = model

    def train_neural_network(self):
        """Train a simple neural network using TensorFlow and print accuracy."""
        # Prepare features and labels
        X = self.df[self.feature_names].values
        y = self.df['target'].values
        # One-hot encode the labels for neural network
        y_categorical = keras.utils.to_categorical(y, num_classes=len(self.target_names))
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
        # Define a simple neural network model
        model = keras.Sequential([
            layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(16, activation='relu'),
            layers.Dense(len(self.target_names), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Neural network accuracy: {accuracy:.2f}")
        # Save the trained model
        self.nn_model = model

    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """Generate a continuation of a given prompt using the GPT-2 model.

        Parameters
        ----------
        prompt : str
            The initial text prompt.
        max_length : int
            Maximum length of the generated sequence.

        Returns
        -------
        str
            The generated text.
        """
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        # Generate text from the model
        with torch.no_grad():
                    output = self.text_model.generate(

                input_ids=input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        # Decode the output tokens back to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

if __name__ == "__main__":
    # Example usage of the AllInOneExample class
    example = AllInOneExample()
    # Summarize the dataset
    example.summarize()
    # Train and evaluate a logistic regression model
    example.train_logistic_regression()
    # Train and evaluate a neural network
    example.train_neural_network()
    # Example text generation
    prompt_text = "Once upon a time,"
    generated = example.generate_text(prompt_text, max_length=30)
    print("\nPrompt:", prompt_text)
    print("Generated text:", generated)
