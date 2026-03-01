from fastapi import FastAPI
import numpy as np
import joblib
import scipy.signal as signal
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# ---------- FIREBASE INIT ----------
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------- LOAD MODELS ----------
freq_model = joblib.load("models/frequency_model.pkl")
mode_model = joblib.load("models/mode_model.pkl")
intensity_model = joblib.load("models/intensity_model.pkl")
le = joblib.load("models/mode_label_encoder.pkl")

app = FastAPI()

# ---------- EMG BUFFER ----------
emg_buffer = []
FS = 1000          # Sampling frequency
WINDOW = 1000      # Samples per window

@app.post("/emg")
def receive_emg(data: dict):
    global emg_buffer

    emg_buffer.append(data["emg"])

    if len(emg_buffer) >= WINDOW:
        emg = np.array(emg_buffer)
        emg_buffer.clear()

        # ---- FEATURE EXTRACTION ----
        rms = np.sqrt(np.mean(emg**2))
        mav = np.mean(np.abs(emg))
        freqs, psd = signal.welch(emg, fs=FS)
        mnf = np.sum(freqs * psd) / np.sum(psd)

        pain_before = data.get("pain_before", 7)

        X = np.array([[rms, mav, mnf, pain_before]])

        # ---- ML INFERENCE ----
        frequency = int(freq_model.predict(X)[0])
        mode = le.inverse_transform(mode_model.predict(X))[0]
        intensity = intensity_model.predict(X)[0].tolist()

        # ---- STORE TO FIREBASE ----
        record = {
            "rms": rms,
            "mav": mav,
            "mnf": mnf,
            "frequency": frequency,
            "mode": mode,
            "intensity": intensity,
            "timestamp": datetime.utcnow()
        }

        db.collection("sessions").add(record)

        return record

    return {"status": "collecting"}