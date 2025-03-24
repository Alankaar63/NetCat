from flask import Flask, request, jsonify
import logging
from ssp import predict_signal_strength
from nfp import predict_network_failure

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Network Prediction API is running!"})

@app.route("/predict/ssp", methods=["POST"])
def predict_ssp():
    """API for Signal Strength Prediction"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        prediction = predict_signal_strength(data)
        return jsonify({"signal_strength_prediction": prediction})

    except Exception as e:
        logging.error(f"SSP Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict/nfp", methods=["POST"])
def predict_nfp():
    """API for Network Failure Prediction"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        prediction = predict_network_failure(data)
        return jsonify({"network_failure_prediction": int(prediction)})

    except Exception as e:
        logging.error(f"NFP Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logging.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
