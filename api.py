from fastapi import FastAPI
import numpy as np
import joblib
from datetime import datetime

app = FastAPI()

# ---------- LOAD MODELS ----------
freq_model = joblib.load("models/frequency_model.pkl")
mode_model = joblib.load("models/mode_model.pkl")
intensity_model = joblib.load("models/intensity_model.pkl")
le = joblib.load("models/mode_label_encoder.pkl")

@app.post("/emg")
def receive_emg(data: dict):
    """
    Expected JSON from Arduino/Python:
    {
      "patient_id": "12",
      "rms": 345.9,
      "mav": 345.9,
      "mnf": 5.37
    }
    """

    rms = data["rms"]
    mav = data["mav"]
    mnf = data["mnf"]

    # Dummy pain score for now (can be replaced later)
    pain_before = 3

    X = np.array([[rms, mav, mnf, pain_before]])

    frequency = int(freq_model.predict(X)[0])
    mode = le.inverse_transform(mode_model.predict(X))[0]
    intensity = int(intensity_model.predict(X)[0])

    response = {
        "frequency_hz": frequency,
        "intensity_mA": intensity,
        "pulse_width_us": 200,
        "mode": mode,
        "timestamp": datetime.utcnow().isoformat()
    }

    return response
