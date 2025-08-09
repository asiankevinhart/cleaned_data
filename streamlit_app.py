import pandas as pd
import numpy as np

# Create date range for one month
dates = pd.date_range(start='2024-06-01', end='2024-06-30')

# Generate normal output_kwh values around 4000 with some noise
np.random.seed(42)
normal_values = np.random.normal(loc=4000, scale=200, size=len(dates))

# Inject some anomalies (very low or very high values)
anomalies_idx = [4, 10, 20]  # arbitrary days with anomalies
for i in anomalies_idx:
    normal_values[i] = normal_values[i] * np.random.choice([0.4, 1.6])  # dip or spike

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'output_kwh': normal_values
})

# Save to CSV
df.to_csv("cleaned_data.csv", index=False)
print("Sample cleaned_data.csv created!")
