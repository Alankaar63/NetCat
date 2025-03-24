# backend/app.py
from flask import Flask, request, jsonify
from inference import InferenceEngine

app = Flask(__name__)
engine = InferenceEngine(model_path='./model.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        throughput = float(request.args.get('throughput', 10))  # Default value if not provided
        latency = float(request.args.get('latency', 100))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid input parameters'}), 400

    features = {
        'Latitude': lat,
        'Longitude': lon,
        'Data Throughput (Mbps)': throughput,
        'Latency (ms)': latency
    }
    prediction = engine.predict(features)
    return jsonify({
        'Latitude': lat,
        'Longitude': lon,
        'Predicted Signal Strength (dBm)': prediction
    })

@app.route('/heatmap', methods=['GET'])
def heatmap():
    try:
        min_lat = float(request.args.get('min_lat'))
        max_lat = float(request.args.get('max_lat'))
        min_lon = float(request.args.get('min_lon'))
        max_lon = float(request.args.get('max_lon'))
        grid_size = int(request.args.get('grid_size', 50))
        throughput = float(request.args.get('throughput', 10))
        latency = float(request.args.get('latency', 100))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid input parameters'}), 400

    heatmap_df = engine.generate_heatmap_grid(
        lat_range=(min_lat, max_lat),
        lon_range=(min_lon, max_lon),
        grid_size=grid_size,
        fixed_throughput=throughput,
        fixed_latency=latency
    )
    # Convert the DataFrame to a list of dictionaries
    heatmap_data = heatmap_df.to_dict(orient='records')
    return jsonify(heatmap_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
