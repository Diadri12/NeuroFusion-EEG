"""
Dependencies for API endpoints
Provides preloaded model and scaler
"""

from deployment.final_inference import load_model, load_scaler

# Preload once at startup
MODEL = load_model()
SCALER = load_scaler()


def get_model_and_scaler():
    """
    Returns preloaded model and scaler for API endpoints.
    """
    return MODEL, SCALER
