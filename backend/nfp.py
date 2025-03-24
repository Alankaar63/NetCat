import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class NFPModelTrainer:
    def __init__(self, data_path='./backend/data/raw.csv', model_path='./backend/models/nfp_model.pkl', 
                 encoder_path='./backend/models/ohe.pkl', feature_path='./backend/models/nfp_features.pkl'):
        self.data_path = os.path.abspath(data_path)
        self.model_path = os.path.abspath(model_path)
        self.encoder_path = os.path.abspath(encoder_path)
        self.feature_path = os.path.abspath(feature_path)

    def load_data(self):
        """Load and preprocess dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)

        # Create 'Failure' column
        df['Failure'] = (
            (df['Signal Strength (dBm)'] < -93) |
            (
                (df['BB60C Measurement (dBm)'] < -93) &
                (df['srsRAN Measurement (dBm)'] < -93) &
                (df['BladeRFxA9 Measurement (dBm)'] < -93) &
                (df['Latency (ms)'] > 90)
            )
        ).astype(int)

        # Convert date columns to timestamps
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_datetime(df[col]).astype(int) // 10**9  # Convert to seconds
            except Exception as e:
                logging.warning(f"Skipping column {col}: {e}")

        # Handle missing values in 'Network Type'
        df['Network Type'] = df['Network Type'].fillna('Unknown')

        return df

    def preprocess_data(self, df):
        """One-hot encode categorical features and prepare dataset."""
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_nt = ohe.fit_transform(df[['Network Type']])
        encoded_nt_df = pd.DataFrame(encoded_nt, columns=ohe.get_feature_names_out(['Network Type']))
        
        df = pd.concat([df, encoded_nt_df], axis=1).drop(columns=['Network Type'], errors='ignore')

        # Drop unnecessary columns
        drop_cols = ['Sr.No.', 'Locality']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        # Define selected features
        selected_features = ["Signal Strength (dBm)", "BB60C Measurement (dBm)", "srsRAN Measurement (dBm)", 
                             "BladeRFxA9 Measurement (dBm)", "Latency (ms)"] + list(encoded_nt_df.columns)

        joblib.dump(ohe, self.encoder_path)
        joblib.dump(selected_features, self.feature_path)

        return df[selected_features], df['Failure']

    def train_model(self, X, y):
        """Train and evaluate the RandomForest model."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        logging.info("Model Performance:")
        logging.info(classification_report(y_test, y_pred))
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        # Save the model
        joblib.dump(model, self.model_path)
        return model

    def train(self):
        """Execute full pipeline: Load, preprocess, train, and save."""
        df = self.load_data()
        X, y = self.preprocess_data(df)
        return self.train_model(X, y)

class NFPModelPredictor:
    def __init__(self, model_path='./backend/models/nfp_model.pkl', encoder_path='./backend/models/ohe.pkl',
                 feature_path='./backend/models/nfp_features.pkl'):
        self.model_path = os.path.abspath(model_path)
        self.encoder_path = os.path.abspath(encoder_path)
        self.feature_path = os.path.abspath(feature_path)

        # Load or train model
        if not os.path.exists(self.model_path):
            logging.warning("Model not found. Training a new model...")
            trainer = NFPModelTrainer()
            trainer.train()

        # Load model, encoder, and feature set
        self.model = joblib.load(self.model_path)
        self.ohe = joblib.load(self.encoder_path)
        self.trained_features = joblib.load(self.feature_path)

    def predict(self, data):
        """Predict network failure for new input data."""
        df = pd.DataFrame([data])
        
        # Handle missing 'Network Type'
        df['Network Type'] = df.get('Network Type', 'Unknown')

        # Apply one-hot encoding
        encoded_nt = self.ohe.transform(df[['Network Type']])
        encoded_nt_df = pd.DataFrame(encoded_nt, columns=self.ohe.get_feature_names_out(['Network Type']))

        # Ensure all training features exist
        for feature in self.trained_features:
            if feature not in encoded_nt_df.columns:
                encoded_nt_df[feature] = 0  

        df = pd.concat([df, encoded_nt_df], axis=1).drop(columns=['Network Type'], errors='ignore')
        df = df.loc[:, ~df.columns.duplicated()].reindex(columns=self.trained_features, fill_value=0)

        # Predict
        prediction = self.model.predict(df)
        return int(prediction[0])


if __name__ == "__main__":
    trainer = NFPModelTrainer()
    trainer.train()

def predict_network_failure(data):
    predictor = NFPModelPredictor()
    return predictor.predict(data)