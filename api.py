from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# ---------- LOAD MODELS ----------
freq_model = joblib.load("models/frequency_model.pkl")
intensity_model = joblib.load("models/intensity_model.pkl")

# ---------- REQUEST SCHEMA ----------
class EMGInput(BaseModel):
    patient_id: str
    rms: float
    mav: float
    mnf: float

# ---------- API ----------
@app.post("/emg")
def receive_emg(data: EMGInput):

    X = np.array([[data.rms, data.mav, data.mnf]])

    frequency = int(freq_model.predict(X)[0])
    intensity = int(intensity_model.predict(X)[0])
    pulse_width = 200  # fixed / rule-based

    return {
        "patient_id": data.patient_id,
        "frequency_hz": frequency,
        "intensity_mA": intensity,
        "pulse_width_us": pulse_width
    }
