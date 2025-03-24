# backend/inference.py
import numpy as np
import pandas as pd
from utils import ModelLoader

class InferenceEngine:
    def __init__(self, model_path='./model.pkl'):
        self.model = ModelLoader.load_model(model_path)

    def predict(self, features: dict):
        """
        Expects a dict with keys:
          'Latitude', 'Longitude', 'Data Throughput (Mbps)', 'Latency (ms)'
        Returns the predicted Signal Strength (dBm).
        """
        df = pd.DataFrame([features])
        prediction = self.model.predict(df)[0]
        return prediction

    def generate_heatmap_grid(self, lat_range, lon_range, grid_size=50,
                              fixed_throughput=10, fixed_latency=100):
        """
        Generates a grid of predictions over the specified region.
        
        Parameters:
            lat_range: tuple (min_lat, max_lat)
            lon_range: tuple (min_lon, max_lon)
            grid_size: number of points per axis
            fixed_throughput: default throughput value for predictions
            fixed_latency: default latency value for predictions
        
        Returns:
            DataFrame with Latitude, Longitude, and Predicted Signal Strength.
        """
        lats = np.linspace(lat_range[0], lat_range[1], grid_size)
        lons = np.linspace(lon_range[0], lon_range[1], grid_size)
        heatmap_data = []

        for lat in lats:
            for lon in lons:
                features = {
                    'Latitude': lat,
                    'Longitude': lon,
                    'Data Throughput (Mbps)': fixed_throughput,
                    'Latency (ms)': fixed_latency
                }
                pred = self.predict(features)
                heatmap_data.append({
                    'Latitude': lat,
                    'Longitude': lon,
                    'Predicted Signal Strength (dBm)': pred
                })
        return pd.DataFrame(heatmap_data)

if __name__ == "__main__":
    engine = InferenceEngine(model_path='./model.pkl')
    # Example prediction:
    features = {
        'Latitude': 25.6,
        'Longitude': 85.1,
        'Data Throughput (Mbps)': 10,
        'Latency (ms)': 100
    }
    print("Predicted Signal Strength:", engine.predict(features))
