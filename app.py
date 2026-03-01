import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import time

# ---------- FIREBASE ----------
cred = credentials.Certificate("firebase_key.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

st.set_page_config(layout="wide")
st.title("Live AI-Based TENS Recommendation Dashboard")

placeholder = st.empty()

while True:
    docs = db.collection("sessions").order_by(
        "timestamp", direction=firestore.Query.DESCENDING
    ).limit(1).stream()

    for doc in docs:
        data = doc.to_dict()

        with placeholder.container():
            st.subheader("📡 Latest Session")

            col1, col2, col3 = st.columns(3)
            col1.metric("Frequency (Hz)", data["frequency"])
            col2.metric("Mode", data["mode"])
            col3.metric("Avg Intensity", round(sum(data["intensity"]) / 4, 2))

            st.write("**RMS:**", round(data["rms"], 3))
            st.write("**MAV:**", round(data["mav"], 3))
            st.write("**MNF:**", round(data["mnf"], 2))
            st.write("**Channel Intensities:**", data["intensity"])

    time.sleep(3)