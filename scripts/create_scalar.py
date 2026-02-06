"""
create_scaler.py
Creates and saves the EEG feature StandardScaler
for the NeuroFusion-EEG deployment pipeline.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import importlib.util

FEATURE_EXTRACTION_PATH = Path(
    r"E:\Diadri\IIT\Year 05 - PT\Final Year Project (FYP)\NeuroFusion-EEG\deployment\model\feature_extraction.py"
)

if not FEATURE_EXTRACTION_PATH.exists():
    raise FileNotFoundError(
        f"feature_extraction.py not found at:\n{FEATURE_EXTRACTION_PATH}"
    )

#  Dynamically import feature_extraction.py
spec = importlib.util.spec_from_file_location(
    "feature_extraction",
    FEATURE_EXTRACTION_PATH
)

feature_extraction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_extraction)

extract_features = feature_extraction.extract_features

print("feature_extraction.py loaded successfully")

# Generate sample EEG signals=
NUM_SAMPLES = 10
SIGNAL_LENGTH = 256

signals = [np.random.randn(SIGNAL_LENGTH) for _ in range(NUM_SAMPLES)]

# Extract features
feature_list = []

for i, signal in enumerate(signals, start=1):
    features = extract_features(signal)

    if features is None or len(features) == 0:
        raise ValueError(f"Feature extraction failed for sample {i}")

    feature_list.append(features)

X = np.array(feature_list)

print(" Feature matrix shape:", X.shape)

# Fit the StandardScaler
scaler = StandardScaler()
scaler.fit(X)

print(" Scaler fitted successfully")
print("   Mean shape:", scaler.mean_.shape)
print("   Std shape :", scaler.scale_.shape)

# Save scaler to deployment/model=
MODEL_DIR = Path(
    r"E:\Diadri\IIT\Year 05 - PT\Final Year Project (FYP)\NeuroFusion-EEG\deployment\model"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"

joblib.dump(scaler, SCALER_PATH)

print(f"Scaler saved at:\n{SCALER_PATH}")

# Test loading the scaler
loaded_scaler = joblib.load(SCALER_PATH)

print(" Scaler reloaded successfully")
print("   Reloaded mean shape:", loaded_scaler.mean_.shape)

print("\n create_scaler.py completed successfully")




