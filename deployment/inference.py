import json
import pickle
import torch
import numpy as np

from .model.model_architecture import DualBranchModel
from .model.feature_extraction import extract_features


class EEGInferenceEngine:
    def __init__(self, model_dir="deployment/model", device=None):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load metadata
        with open(f"{model_dir}/model_metadata.json", "r") as f:
            metadata = json.load(f)

        self.signal_length = metadata["signal_length"]
        self.n_features = metadata["n_features"]
        self.n_classes = metadata["n_classes"]
        self.class_labels = metadata.get("class_labels", None)

        # Load scaler
        with open(f"{model_dir}/feature_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        # Load model
        self.model = DualBranchModel(
            signal_length=self.signal_length,
            n_features=self.n_features,
            n_classes=self.n_classes
        )

        self.model.load_state_dict(
            torch.load(f"{model_dir}/model_weights.pt", map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, eeg_signal: np.ndarray):
        """
        eeg_signal shape: (signal_length,)
        """

        # Branch A (raw signal)
        signal_tensor = torch.tensor(
            eeg_signal, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, signal_length)
        signal_tensor = signal_tensor.to(self.device)

        # Branch B (handcrafted features)
        features = extract_features(eeg_signal)
        features = self.scaler.transform([features])

        feature_tensor = torch.tensor(
            features, dtype=torch.float32
        ).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(signal_tensor, feature_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)

        if self.class_labels:
            return {
                "prediction": self.class_labels[pred_idx],
                "confidence": float(probs[pred_idx]),
                "probabilities": probs.tolist()
            }

        return {
            "prediction_index": int(pred_idx),
            "confidence": float(probs[pred_idx]),
            "probabilities": probs.tolist()
        }

