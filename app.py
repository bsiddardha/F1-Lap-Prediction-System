from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("f1_random_forest.pkl")
feature_columns = joblib.load("f1_feature_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs
        data = {
            "Year": int(request.form["year"]),
            "LapNumber": int(request.form["lap_number"]),
            "Stint": int(request.form["stint"]),
            "TyreLife": int(request.form["tyre_life"]),
            "FreshTyre": int(request.form["fresh_tyre"]),
            "Position": int(request.form["position"]),
            "Compound": request.form["compound"].upper().strip(),
            "Team": request.form["team"].strip(),
            "Driver": request.form["driver"].upper().strip()
        }

        input_df = pd.DataFrame([data])

        # One-hot encoding
        input_df = pd.get_dummies(
            input_df,
            columns=["Compound", "Team", "Driver"]
        )

        # 🔥 Match training columns EXACTLY
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)

        return render_template(
            "index.html",
            prediction_text=f"{round(prediction[0],3)} seconds"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
