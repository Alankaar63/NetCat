import numpy as np
import pandas as pd
import joblib
import subprocess  # To execute train_model.py
from utils import ModelLoader


# Function to retrain the model when requested
def retrain_model():
    print("Retraining model before inference...")
    subprocess.run(["python", "./backend/train_model.py"], check=True)  # Runs train_model.py


class InferenceEngine:
    def __init__(self, model_path='./backend/models/ssp_model.pkl'):
        self.model_path = model_path
        self.model = ModelLoader.load_model(model_path)  # Load the model initially

    def predict(self, features: dict):
        """
        Parameters:
            features (dict): Dictionary with required features for prediction.
                             Must include 'retrain' key (0 or 1).
        
        Returns:
            float: Predicted Signal Strength (dBm)
        """
        retrain_flag = features.pop('retrain', 0)  # Extract retrain flag, default to 0
        
        if retrain_flag == 1:
            retrain_model()  # Retrain only if requested
            self.model = ModelLoader.load_model(self.model_path)  # Reload the updated model

        df = pd.DataFrame([features])
        prediction = self.model.predict(df)[0]
        return prediction

    def generate_heatmap_grid(self, lat_range, lon_range, grid_size=50,
                              fixed_throughput=10, fixed_latency=100, retrain=0):
        """
        Generates a grid of predictions over the specified region.
        
        Parameters:
            lat_range: tuple (min_lat, max_lat)
            lon_range: tuple (min_lon, max_lon)
            grid_size: number of points per axis
            fixed_throughput: default throughput value for predictions
            fixed_latency: default latency value for predictions
            retrain: 1 to retrain the model before generating heatmap, 0 to use the existing model

        Returns:
            DataFrame with Latitude, Longitude, and Predicted Signal Strength.
        """
        if retrain == 1:
            retrain_model()  # Retrain only if requested
            self.model = ModelLoader.load_model(self.model_path)  # Reload the updated model

        lats = np.linspace(lat_range[0], lat_range[1], grid_size)
        lons = np.linspace(lon_range[0], lon_range[1], grid_size)
        heatmap_data = []

        for lat in lats:
            for lon in lons:
                features = {
                    'Latitude': lat,
                    'Longitude': lon,
                    'Data Throughput (Mbps)': fixed_throughput,
                    'Latency (ms)': fixed_latency,
                    'retrain': 0  # Ensure retrain is explicitly set for each call
                }
                pred = self.predict(features)
                heatmap_data.append({
                    'Latitude': lat,
                    'Longitude': lon,
                    'Predicted Signal Strength (dBm)': pred
                })
        return pd.DataFrame(heatmap_data)


if __name__ == "__main__":
    engine = InferenceEngine(model_path='./backend/models/ssp_model.pkl')
    
    # Example prediction with retraining disabled
    features = {
        'Latitude': 25.6,
        'Longitude': 85.1,
        'Data Throughput (Mbps)': 10,
        'Latency (ms)': 100,
        'retrain': 1  # Change to 0 if you don't want to retrain
    }
    print("Predicted Signal Strength:", engine.predict(features), "dBm")

engine = InferenceEngine()

def predict_signal_strength(features):
    return engine.predict(features)