import pandas as pd
import numpy as np
import os

BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATASET_1  = os.path.join(BASE_DIR, "datasets", "epilepsy_diagnosis", "EEG_Signal.csv")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

WINDOW_SIZE = 256
STRIDE      = 20
N_WINDOWS   = 200  # ~4400 rows needed

# Bonn EEG label mapping
SEIZURE_LABEL    = 'E'   # Ictal — active seizure
NO_SEIZURE_LABEL = 'A'   # Normal surface EEG (closest to Interictal for API)

def save_signal_csv(samples: np.ndarray, path: str):
    pd.DataFrame({"Signal": samples.astype(np.float64)}).to_csv(path, index=False)
    n_windows = (len(samples) - WINDOW_SIZE) // STRIDE + 1
    print(f"  Saved : {os.path.basename(path)}")
    print(f"          {len(samples):,} rows -> ~{n_windows} windows for API")

def extract_segment(series: pd.Series, n_windows: int) -> np.ndarray:
    needed = n_windows * STRIDE + WINDOW_SIZE
    values = series.dropna().values.astype(np.float64)
    if len(values) < needed:
        values = np.tile(values, int(np.ceil(needed / len(values))) + 1)
    return values[:needed]

def main():
    print(f"Loading: {DATASET_1}\n")
    df = pd.read_csv(DATASET_1)
    print(f"  Shape        : {df.shape}")
    print(f"  Label counts : {df['Labels'].value_counts().to_dict()}\n")

    # Seizure (Label E)
    seizure_rows = df[df['Labels'] == SEIZURE_LABEL]['Signal']
    print(f"Seizure rows    (E - Ictal)      : {len(seizure_rows):,}")
    seizure_signal = extract_segment(seizure_rows, N_WINDOWS)
    print(f"  -> Extracted {len(seizure_signal):,} samples\n")

    # No-Seizure (Label A)
    no_seizure_rows = df[df['Labels'] == NO_SEIZURE_LABEL]['Signal']
    print(f"No-seizure rows (A - Normal EEG) : {len(no_seizure_rows):,}")
    no_seizure_signal = extract_segment(no_seizure_rows, N_WINDOWS)
    print(f"  -> Extracted {len(no_seizure_signal):,} samples\n")

    # Save
    print("Saving CSVs")
    save_signal_csv(seizure_signal,    os.path.join(OUTPUT_DIR, "seizure_eeg.csv"))
    save_signal_csv(no_seizure_signal, os.path.join(OUTPUT_DIR, "no_seizure_eeg.csv"))

    print("\n Done")
    print("Upload both CSVs to EpiGuard to test seizure / no-seizure detection.")
    print(f"API column: 'Signal' | window={WINDOW_SIZE} | stride={STRIDE}")

if __name__ == "__main__":
    main()