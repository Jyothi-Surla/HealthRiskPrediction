import pandas as pd

# Load dataset
df = pd.read_csv("heart.csv")

# Show all columns
print("\n📋 All columns:")
print(df.columns.tolist())

# Check target value distribution
print("\n🎯 Target value counts:")
print(df["target"].value_counts())

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Confirm shape
print(f"\n✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
