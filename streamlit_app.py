import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.title("⚡ Energy AI Dashboard")

uploaded = st.file_uploader("Upload Energy Data (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Basic validation
    if 'date' not in df.columns or 'output_kwh' not in df.columns:
        st.error("CSV must contain 'date' and 'output_kwh' columns.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'])

    st.subheader("Energy Output Over Time")
    st.line_chart(df.set_index('date')['output_kwh'])

    # Anomaly detection
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(df[['output_kwh']]) == -1
    anomalies = df[df['anomaly']]

    # Dynamic summary
    if not anomalies.empty:
        anomaly_dates = ", ".join(anomalies['date'].dt.strftime('%b %d').tolist())
    else:
        anomaly_dates = "None"

    avg_kwh = df['output_kwh'].mean()
    peak_kwh = df['output_kwh'].max()
    summary = f"Avg = {avg_kwh:.1f} kWh, Anomalies = {anomaly_dates}, Peak = {peak_kwh:.1f} kWh"

    st.subheader("Weekly Summary")
    st.markdown(f"**{summary}**")

    # Visualization with anomalies highlighted
    st.subheader("Energy Output with Anomalies")
    fig, ax = plt.subplots()
    ax.plot(df['date'], df['output_kwh'], label='Output (kWh)')
    if not anomalies.empty:
        ax.scatter(anomalies['date'], anomalies['output_kwh'], color='red', label='Anomaly')
    ax.set_xlabel("Date")
    ax.set_ylabel("Output (kWh)")
    ax.set_title("Energy Output and Anomalies")
    ax.legend()
    st.pyplot(fig)

    # Anomaly table
    st.subheader("Detected Anomalies")
    if not anomalies.empty:
        st.write(anomalies[['date', 'output_kwh']])
    else:
        st.write("No anomalies detected.")

else:
    st.info("Please upload a CSV file with 'date' and 'output_kwh' columns to get started.")
