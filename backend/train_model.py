import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from utils import DataPreprocessor, ModelSaver

class ModelTrainer:
    def __init__(self, data_path, model_path='./backend/models/ssp_model.pkl'):
        self.data_path = os.path.abspath(data_path)  # Convert to absolute path
        self.model_path = os.path.abspath(model_path)  # Convert to absolute path

        # Check if data file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def train(self):
        print(f"Loading data from: {self.data_path}")

        # Load and preprocess data
        preprocessor = DataPreprocessor(self.data_path)
        df = preprocessor.load_data()
        df = preprocessor.preprocess(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)

        # Define individual models
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
        et = ExtraTreesRegressor(n_estimators=50, random_state=42)

        # Combine models into a VotingRegressor ensemble
        ensemble = VotingRegressor([('rf', rf), ('gb', gb), ('et', et)])
        
        print("Training ensemble model...")
        ensemble.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = ensemble.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"✅ Ensemble model trained. RMSE on test set: {rmse:.4f}")

        # Save the trained model
        ModelSaver.save_model(ensemble, self.model_path)
        print(f"✅ Model saved to {self.model_path}")

        return ensemble

if __name__ == "__main__":
    trainer = ModelTrainer(data_path='./backend/data/raw.csv', model_path='./backend/models/ssp_model.pkl')
    trainer.train()
