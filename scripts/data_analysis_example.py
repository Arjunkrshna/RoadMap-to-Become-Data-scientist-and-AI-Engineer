#!/usr/bin/env python3
"""
Data Analysis Example

This script demonstrates a simple data analysis workflow using pandas and matplotlib.

We define a class `DataAnalysisExample` that loads a DataFrame, computes summary statistics,
and plots a histogram. The example usage at the bottom shows how to use the class.
"""
import pandas as pd
import matplotlib.pyplot as plt

class DataAnalysisExample:
    def __init__(self):
        """Initialize the class with an empty DataFrame."""
        self.df = pd.DataFrame()

    def load_data(self, data: dict):
        """
        Load data into a pandas DataFrame.

        Parameters
        ----------
        data : dict
            A dictionary where keys are column names and values are lists of data.
            For example: {"age": [25, 30, 22], "salary": [50000, 60000, 45000]}
        """
        self.df = pd.DataFrame(data)
        print("Data loaded successfully.")
        print(self.df.head())

    def summary(self):
        """Print summary statistics of the DataFrame."""
        if self.df.empty:
            print("No data loaded.")
            return
        print("Summary statistics:")
        print(self.df.describe())

    def plot_histogram(self, column: str):
        """
        Plot a histogram for a given column of the DataFrame.

        Parameters
        ----------
        column : str
            The name of the column to plot.
        """
        if column not in self.df.columns:
            print(f"Column '{column}' not found in data.")
            return
        self.df[column].plot(kind="hist", bins=10, title=f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()


if __name__ == "__main__":
    # Example usage
    data = {
        "age": [25, 30, 22, 35, 40, 28],
        "salary": [50000, 60000, 45000, 80000, 120000, 65000]
    }

    analysis = DataAnalysisExample()
    analysis.load_data(data)    # Load the data into the DataFrame
    analysis.summary()          # Print summary statistics
    analysis.plot_histogram("age")  # Plot a histogram of the 'age' column
