import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
print("Loading dataset...")
df = pd.read_csv("bahrain_2021_2024_combined.csv")

# Features and target
features = [
    "Year",
    "LapNumber",
    "Stint",
    "TyreLife",
    "FreshTyre",
    "Position",
    "Compound",
    "Team",
    "Driver"
]

target = "LapTimeSeconds"

# Clean data
print("Cleaning data...")
df = df[features + [target]].dropna()

# Split features and target
X = df[features]
y = df[target]

# One-hot encode categorical columns
print("Encoding categorical features...")
X = pd.get_dummies(
    X,
    columns=["Compound", "Team", "Driver"]
)

# Convert all columns to numeric
X = X.astype(float)

# Save feature columns
feature_columns = X.columns.tolist()

print("Dataset shape:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Simple Random Forest
print("Training model...")
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
print("Making predictions...")
preds = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, preds)

print("\n✅ Model trained successfully!")
print(f"📊 MAE: {mae:.3f} seconds")

# Save model and columns
joblib.dump(model, "f1_random_forest.pkl")
joblib.dump(feature_columns, "f1_random_forest_columns.pkl")

print("💾 Model saved as: f1_random_forest.pkl")
print("💾 Columns saved as: f1_random_forest_columns.pkl")