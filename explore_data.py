import pandas as pd

# Load the dataset
df = pd.read_csv("heart.csv")

# Show the first 5 rows
print("\n🔍 First 5 rows of the dataset:")
print(df.head())

# Show basic info (column names, data types, nulls)
print("\n📊 Dataset info:")
print(df.info())

# Show basic statistics (mean, std, min, max)
print("\n📈 Statistical summary:")
print(df.describe())
