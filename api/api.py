from fastapi import FastAPI, File, UploadFile, HTTPException
from api.dependencies import get_model_and_scaler
from deployment.final_inference import run_inference
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# For MAT files
from scipy.io import loadmat

# For EDF files
import mne
import io

app = FastAPI(title="NeuroFusion EEG Inference API", version="1.0")

# ADD CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict-file")
async def predict_eeg_file(file: UploadFile = File(...)):
    """
    Endpoint to predict seizure class from EEG file (CSV, EDF, TXT, MAT)
    """
    try:
        content = await file.read()
        extension = file.filename.split(".")[-1].lower()

        # CSV: load with pandas
        if extension == "csv":
            df = pd.read_csv(io.BytesIO(content))
            signal = df.values.flatten().tolist()
        
         # EXCEL
        elif extension in ["xlsx", "xls"]:
            df = pd.read_excel(io.BytesIO(content))
            signal = df.select_dtypes(include=[np.number]).values.flatten().tolist()

        # TXT: assume whitespace or comma separated numbers
        elif extension == "txt":
            text_data = content.decode("utf-8")
            # split by whitespace or commas
            nums = text_data.replace(",", " ").split()
            signal = [float(x) for x in nums]

        # MAT: MATLAB format
        elif extension == "mat":
            mat_data = loadmat(io.BytesIO(content))
            # Attempt to find first array in the MAT file
            signal_array = None
            for key in mat_data:
                if not key.startswith("__") and isinstance(mat_data[key], np.ndarray):
                    signal_array = mat_data[key].flatten()
                    break
            if signal_array is None:
                raise HTTPException(status_code=400, detail="No valid array found in MAT file")
            signal = signal_array.tolist()

        # EDF: EEG format
        elif extension == "edf":
            # Load EDF using MNE
            with io.BytesIO(content) as f:
                raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                data, _ = raw[:]
                signal = data.flatten().tolist()

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Get model & scaler
        model, scaler = get_model_and_scaler()

        # Run inference
        result = run_inference(signal=signal, model_in=model, scaler_in=scaler)

        # Map numeric classes to labels
        label_map = {0: "Normal", 1: "Interictal", 2: "Seizure"}
        confidence = round(result["pred_prob"] * 100, 2)

        return {
            "pred_class": result["pred_class"],
            "label": label_map[result["pred_class"]],
            "confidence": confidence,
            "pred_prob": result["pred_prob"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
