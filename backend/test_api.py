# backend/test_api.py
import unittest
import json
from app import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        # Test /predict with valid parameters
        response = self.app.get('/predict?lat=25.6&lon=85.1&throughput=10&latency=100')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('Predicted Signal Strength (dBm)', data)

    def test_heatmap_endpoint(self):
        # Test /heatmap with valid bounding box parameters
        response = self.app.get('/heatmap?min_lat=25.4&max_lat=25.8&min_lon=84.9&max_lon=85.3&grid_size=10')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIsInstance(data, list)
        if data:
            self.assertIn('Predicted Signal Strength (dBm)', data[0])

if __name__ == '__main__':
    unittest.main()
