# app.py
from flask import Flask, request, jsonify
from api import predict_tens_params

app = Flask(__name__)

@app.route("/emg", methods=["POST"])
def predict():
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
