import pandas as pd
import numpy as np

# Generate date range
dates = pd.date_range(start="2025-06-01", periods=60, freq="D")

# Simulate kWh output with some random noise
np.random.seed(42)
output_kwh = np.random.normal(loc=4500, scale=300, size=len(dates))

# Introduce some anomalies (very high or low values)
output_kwh[10] = 7000
output_kwh[25] = 2000
output_kwh[40] = 6500

# Create DataFrame
df = pd.DataFrame({
    "date": dates,
    "output_kwh": output_kwh
})

# Save to CSV
df.to_csv("cleaned_data.csv", index=False)
print("✅ cleaned_data.csv created with 60 days of sample data.")
