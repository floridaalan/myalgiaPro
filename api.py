# api.py
import pickle
import numpy as np

mode_model = pickle.load(open("models/mode_model.pkl", "rb"))

def predict_tens_params(rms, mav, mnf, feedback=None):
    X = np.array([[rms, mav, mnf]])
    
    mode = mode_model.predict(X)[0]

    # Simple rule-based tuning (example)
    frequency = 80 if mnf < 6 else 100
    intensity = min(30, rms / 15)
    pulse_width = 200

    # Adaptive adjustment using feedback
    if feedback:
        intensity += (feedback - 3) * 2

    return {
        "frequency": round(frequency, 2),
        "intensity": round(intensity, 2),
        "pulse_width": pulse_width,
        "mode": mode
    }
