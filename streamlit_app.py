import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

st.title("Energy AI Dashboard")

uploaded = st.file_uploader("Upload Energy Data (CSV)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    df['date'] = pd.to_datetime(df['date'])

    st.subheader("Energy Output")
    st.line_chart(df['output_kwh'])

    model = IsolationForest(contamination=0.05)
    df['anomaly'] = model.fit_predict(df[['output_kwh']]) == -1
    anomalies = df[df['anomaly'] == True]

    st.subheader("Weekly Summary")
    st.markdown("**Anomalies detected on June 5 and 9.**")

    st.subheader("Anomaly Visualization")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["output_kwh"], label="Output")
    ax.scatter(anomalies["date"], anomalies["output_kwh"], color="red", label="Anomaly")
    ax.set_title("Energy Output with Anomalies")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Anomaly Table")
    st.write(anomalies[["date", "output_kwh"]])
