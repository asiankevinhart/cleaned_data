import pandas as pd

# Load your uploaded file
file_path = '/mnt/data/CSV - Sheet1.csv'
df = pd.read_csv(file_path)

# Show the first few rows and columns
print(df.head())
print(df.columns)
