import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    """
    Loads and preprocesses the raw CSV data.
    Uses a subset of columns to predict signal strength.
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        # Debugging statements
        print("Current working directory:", os.getcwd())  
        print("Looking for file at:", self.filepath)

        # Check if the file exists
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        df = pd.read_csv(self.filepath)
        
        # Convert Timestamp to datetime if it exists
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        return df

    def preprocess(self, df):
        # Drop the Signal Quality column if it exists (all zeros)
        if 'Signal Quality (%)' in df.columns:
            df = df.drop(columns=['Signal Quality (%)'])

        # Select important features
        columns_to_use = ['Latitude', 'Longitude', 'Data Throughput (Mbps)', 'Latency (ms)', 'Signal Strength (dBm)']
        
        # Check if all required columns exist
        missing_columns = [col for col in columns_to_use if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in dataset: {missing_columns}")

        df = df[columns_to_use].dropna()
        return df

    def split_data(self, df, test_size=0.2, random_state=42):
        X = df.drop(columns=['Signal Strength (dBm)'])
        y = df['Signal Strength (dBm)']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

class ModelSaver:
    """Helper to save a model."""
    @staticmethod
    def save_model(model, filepath):
        joblib.dump(model, filepath)

class ModelLoader:
    """Helper to load a saved model."""
    @staticmethod
    def load_model(filepath):
        return joblib.load(filepath)
