import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("bahrain_2021_2024_combined.csv")

# Features
features = [
    "Year", "LapNumber", "Stint", "TyreLife",
    "FreshTyre", "Position", "Compound", "Team", "Driver"
]

# ✅ Correct target
target = "LapTimeSeconds"

# Clean data
df = df[features + [target]].dropna()

# Split
X = df[features]
y = df[target]

# One-hot encoding
X = pd.get_dummies(X, columns=["Compound", "Team", "Driver"])

# Save columns
feature_columns = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("✅ Model trained successfully!")
print(f"📊 MAE: {mae:.3f} seconds")

# Save
joblib.dump(model, "f1_random_forest.pkl")
joblib.dump(feature_columns, "f1_feature_columns.pkl")

print("💾 Model and columns saved!")
