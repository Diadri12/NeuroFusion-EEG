"""
test_api_request.py
This script sends a dummy EEG signal to the NeuroFusion API and prints the prediction.
"""

import requests
import numpy as np

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Create a dummy EEG signal of length 256
dummy_signal = np.random.randn(256).tolist()

# Prepare request payload
payload = {"signal": dummy_signal}

# Send POST request
response = requests.post(url, json=payload)

# Check response
if response.status_code == 200:
    data = response.json()
    print("Prediction successful!")
    print(f"Predicted class: {data['pred_class']}")
    print(f"Predicted probability: {data['pred_prob']:.4f}")
    print(f"Label: {data['label']}")
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)
