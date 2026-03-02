from flask import Flask, request, jsonify
from api import predict_tens_params
import traceback

app = Flask(__name__)

@app.route("/")
def home():
    return "EMG–TENS API running successfully"

@app.route("/emg", methods=["POST"])
def emg():
    try:
        data = request.get_json(force=True)

        result = predict_tens_params(
            data["rms"],
            data["mav"],
            data["mnf"],
            data.get("feedback")
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500
