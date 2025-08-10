import os
import pandas as pd
import streamlit as st
from datetime import datetime

# Create a temporary folder (inside the app directory)
LOCAL_FOLDER = "alerts_temp"
os.makedirs(LOCAL_FOLDER, exist_ok=True)

st.title("Energy Data Anomaly Detection")

uploaded_file = st.file_uploader("Upload Energy Data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.dataframe(df.head())

    # Dummy anomaly detection
    if "output_kwh" in df.columns and "date" in df.columns:
        anomalies = df[df["output_kwh"] > 1000][["date", "output_kwh"]]
    else:
        st.error("CSV missing required columns: 'date' and 'output_kwh'")
        anomalies = pd.DataFrame()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alerts_filename = f"alerts_{timestamp}.csv"
    alerts_path = os.path.join(LOCAL_FOLDER, alerts_filename)

    if anomalies.empty:
        df_to_save = pd.DataFrame({"message": ["No anomalies found"]})
    else:
        df_to_save = anomalies

    df_to_save.to_csv(alerts_path, index=False)

    st.success(f"Alerts saved locally: {alerts_path}")
    st.download_button(
        label="Download Alerts CSV",
        data=df_to_save.to_csv(index=False).encode('utf-8'),
        file_name=alerts_filename,
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to detect anomalies.")
