from flask import Flask, request, jsonify
from api import predict_tens_params

app = Flask(__name__)

@app.route("/")
def home():
    return "EMG–TENS API running successfully"

@app.route("/emg", methods=["POST"])
def emg():
    data = request.get_json(force=True)

    required = ["patient_id", "rms", "mav", "mnf"]
    if not all(k in data for k in required):
        return jsonify({"error": "Invalid input"}), 400

    result = predict_tens_params(
        data["rms"],
        data["mav"],
        data["mnf"],
        data.get("feedback")
    )

    return jsonify(result)
