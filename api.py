import joblib
import numpy as np

mode_model = joblib.load("models/mode_model.pkl")

def predict_tens_params(rms, mav, mnf, feedback=None):

    # 🔑 DEFAULT feedback if first session
    if feedback is None:
        feedback = 3   # neutral pain score

    # 🔑 MODEL EXPECTS 4 FEATURES
    X = np.array([[rms, mav, mnf, feedback]])

    mode = mode_model.predict(X)[0]

    frequency = 80 if mnf < 6 else 100
    intensity = min(30, rms / 15)
    pulse_width = 200

    return {
        "frequency": round(float(frequency), 2),
        "intensity": round(float(intensity), 2),
        "pulse_width": pulse_width,
        "mode": str(mode)
    }
