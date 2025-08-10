import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime
import requests  # for optional Zapier integration
from openai import OpenAI  # if using OpenAI for AI summary

# Create a temporary folder to save alerts CSVs
LOCAL_FOLDER = "alerts_temp"
os.makedirs(LOCAL_FOLDER, exist_ok=True)

st.title("Energy Data Anomaly Detection & AI Dashboard")

uploaded_file = st.file_uploader("Upload Energy Data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure 'date' column exists and is datetime
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

    # Run IsolationForest anomaly detection
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
    else:
        output_message = "No anomalies detected."
        weekly_summary = ""

    st.subheader("Anomaly Summary")
    st.text(output_message)
    st.text(weekly_summary)

    # -----------------------------
    # AI-generated summary section
    # -----------------------------
    if "OPENAI_API_KEY" in os.environ:  # Only if API key is available
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        prompt = f"""
        You are an energy monitoring assistant.
        Summarize the following anomaly detection results for a weekly energy report in plain language.

        Output message: {output_message}
        Weekly summary: {weekly_summary}

        Keep it concise and easy to understand.
        """
        try:
            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            ai_summary = ai_response.choices[0].message.content.strip()
            st.subheader("AI-Generated Summary")
            st.write(ai_summary)
        except Exception as e:
            st.warning(f"AI summary could not be generated: {e}")
    else:
        st.info("Set the OPENAI_API_KEY environment variable to enable AI-generated summaries.")

    # Plot anomalies on the graph
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

    # Save alerts CSV locally and provide download button
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alerts_filename = f"alerts_{timestamp}.csv"
    alerts_path = os.path.join(LOCAL_FOLDER, alerts_filename)

    if anomalies.empty:
        df_to_save = pd.DataFrame({"message": ["No anomalies found"]})
    else:
        df_to_save = anomalies[["date", "output_kwh"]]

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
