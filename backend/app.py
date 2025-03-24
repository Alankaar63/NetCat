from flask import Flask, request, jsonify
from ssp import predict_signal_strength
from nfp import predict_network_failure

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "AI Network Monitoring API is running!"})

@app.route("/predict_signal", methods=["POST"])
def predict_signal():
    data = request.get_json()
    prediction = predict_signal_strength(data)
    return jsonify({"signal_strength": prediction})

@app.route("/predict_failure", methods=["POST"])
def predict_failure():
    data = request.get_json()
    prediction = predict_network_failure(data)
    return jsonify({"network_failure": prediction})

@app.route("/heatmap", methods=["POST"])
def heatmap():
    data = request.get_json()
    signal = predict_signal_strength(data)
    failure = predict_network_failure(data)
    return jsonify({"signal_strength": signal, "network_failure": failure})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
