import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

# ========================
# Step 0: Generate fake data if not found
# ========================
if not os.path.exists("cleaned_data.csv"):
    print("No cleaned_data.csv found. Generating fake dataset...")
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
    output_kwh = np.random.normal(loc=5000, scale=500, size=len(dates))
    data = pd.DataFrame({"date": dates, "output_kwh": output_kwh})
    data.to_csv("cleaned_data.csv", index=False)
    print("✅ Fake cleaned_data.csv created.")

# ========================
# Step 1: Load dataset
# ========================
print("Loading dataset...")
df = pd.read_csv("cleaned_data.csv")
df['date'] = pd.to_datetime(df['date'])

# ========================
# Step 2: Feature engineering
# ========================
print("Engineering features...")
df['rolling_7d'] = df['output_kwh'].rolling(window=7, min_periods=1).mean()
df['weekday'] = df['date'].dt.weekday
df['lag_1'] = df['output_kwh'].shift(1)
df.dropna(inplace=True)

features = ['output_kwh', 'rolling_7d', 'weekday', 'lag_1']

# ========================
# Step 3: Train model
# ========================
print("Training IsolationForest...")
model = IsolationForest(
    contamination=0.03,
    n_estimators=150,
    max_samples='auto',
    random_state=42
)
model.fit(df[features])

# ========================
# Step 4: Predict anomalies
# ========================
print("Predicting anomalies...")
df['anomaly'] = model.predict(df[features]) == -1

# ========================
# Step 5: Save predictions
# ========================
output_df = df[['date', 'output_kwh', 'anomaly']]
output_df.to_csv("predictions.csv", index=False)
print("✅ Predictions saved to predictions.csv")

# ========================
# Step 6: Generate summary
# ========================
avg_kwh = df['output_kwh'].mean()
peak_kwh = df['output_kwh'].max()
peak_date = df.loc[df['output_kwh'].idxmax(), 'date'].strftime('%Y-%m-%d')
anomaly_dates = df.loc[df['anomaly'], 'date'].dt.strftime('%Y-%m-%d').tolist()

summary = (
    f"Summary of site performance:\n"
    f"- Average output: {avg_kwh:.2f} kWh\n"
    f"- Peak output: {peak_kwh:.2f} kWh on {peak_date}\n"
    f"- Anomalies detected on: {', '.join(anomaly_dates) if anomaly_dates else 'None'}"
)

# ========================
# Step 7: Save summary
# ========================
with open("weekly_summary.txt", "w") as f:
    f.write(summary)
print("✅ Summary saved to weekly_summary.txt")

print("\nPipeline complete!")
