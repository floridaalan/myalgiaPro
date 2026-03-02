from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# ---------- FIREBASE ----------
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------- LOAD MODELS ----------
freq_model = joblib.load("models/frequency_model.pkl")
mode_model = joblib.load("models/mode_model.pkl")
intensity_model = joblib.load("models/intensity_model.pkl")
le = joblib.load("models/mode_label_encoder.pkl")

app = FastAPI()

# ---------- INPUT SCHEMA ----------
class EMGFeatures(BaseModel):
    patient_id: str
    rms: float
    mav: float
    mnf: float
    pain_before: int | None = 7

# ---------- ENDPOINT ----------
@app.post("/emg")
def receive_emg(data: EMGFeatures):

    X = np.array([[data.rms, data.mav, data.mnf, data.pain_before]])

    frequency = int(freq_model.predict(X)[0])
    mode = le.inverse_transform(mode_model.predict(X))[0]
    intensity = int(intensity_model.predict(X)[0])

    pulse_width = 200  # fixed / rule-based for now

    record = {
        "patient_id": data.patient_id,
        "rms": data.rms,
        "mav": data.mav,
        "mnf": data.mnf,
        "frequency_hz": frequency,
        "intensity_mA": intensity,
        "pulse_width_us": pulse_width,
        "mode": mode,
        "timestamp": datetime.utcnow()
    }

    db.collection("sessions").add(record)

    return {
        "frequency_hz": frequency,
        "intensity_mA": intensity,
        "pulse_width_us": pulse_width,
        "mode": mode
    }
