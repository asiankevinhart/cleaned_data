import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Path to your Google Drive Zapier watch folder
DRIVE_FOLDER = r"G:\My Drive\Zapier Watch"
os.makedirs(DRIVE_FOLDER, exist_ok=True)

st.title("Energy Data Anomaly Detection & AI Dashboard")

uploaded_file = st.file_uploader("Upload Energy Data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure date column exists and is datetime
    if "date" in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        st.error("CSV must contain a 'date' column.")
        st.stop()

    if "output_kwh" not in df.columns:
        st.error("CSV must contain an 'output_kwh' column.")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Energy Output Over Time")
    st.line_chart(df.set_index('date')['output_kwh'])

    # Run anomaly detection
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(df[['output_kwh']]) == -1
    anomalies = df[df['anomaly']]

    # Prepare anomaly messages
    if not anomalies.empty:
        first_date = anomalies["date"].iloc[0].strftime("%B %d")
        first_value = anomalies["output_kwh"].iloc[0]
        anomaly_dates = anomalies["date"].dt.strftime("%B %d").tolist()

        output_message = f"Output drop on {first_date}\nValue: {first_value} kWh"
        weekly_summary = f"Weekly Summary: Anomalies detected on {', '.join(anomaly_dates)}."

        # Mock AI-generated summary (custom text)
        ai_summary = (
            f"Based on the analysis, unusual drops in energy output were detected on "
            f"{', '.join(anomaly_dates)}. The most significant occurred on {first_date} "
            f"with an output of {first_value} kWh, suggesting potential equipment or "
            f"environmental issues. Further investigation is recommended."
        )

    else:
        output_message = "No anomalies detected."
        weekly_summary = ""
        ai_summary = "AI Summary: Energy output is stable with no significant irregularities."

    st.subheader("Anomaly Summary")
    st.text(output_message)
    st.text(weekly_summary)
    st.text(ai_summary)

    # Visualization
    st.subheader("Anomaly Visualization")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["output_kwh"], label="Output")
    ax.scatter(anomalies["date"], anomalies["output_kwh"], color="red", label="Anomaly")
    ax.set_title("Energy Output with Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Output kWh")
    ax.legend()
    st.pyplot(fig)

    # Show anomalies table
    st.subheader("Anomaly Table")
    st.dataframe(anomalies[["date", "output_kwh"]])

    # Save alerts CSV directly to Google Drive Zapier Watch folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alerts_filename = f"alerts_{timestamp}.csv"
    alerts_path = os.path.join(DRIVE_FOLDER, alerts_filename)

    if anomalies.empty:
        df_to_save = pd.DataFrame({"message": ["No anomalies found"]})
    else:
        df_to_save = anomalies[["date", "output_kwh"]]
        # Append the messages to CSV for Zapier
        df_to_save["output_message"] = output_message
        df_to_save["weekly_summary"] = weekly_summary
        df_to_save["ai_summary"] = ai_summary

    df_to_save.to_csv(alerts_path, index=False)

    st.success(f"Alerts saved to Google Drive Zapier Watch folder: {alerts_path}")

    # Also provide Streamlit download button
    st.download_button(
        label="Download Alerts CSV",
        data=df_to_save.to_csv(index=False).encode('utf-8'),
        file_name=alerts_filename,
        mime='text/csv'
    )

else:
    st.info("Please upload a CSV file to detect anomalies.")
