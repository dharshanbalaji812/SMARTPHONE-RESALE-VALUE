from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
MODEL_PATH = Path("models/resale_model.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found. Run 'python train_model.py' first.")

model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "brand": request.form["brand"],
            "age_years": float(request.form["age_years"]),
            "original_price": float(request.form["original_price"]),
            "ram_gb": int(request.form["ram_gb"]),
            "storage_gb": int(request.form["storage_gb"]),
            "battery_health": int(request.form["battery_health"]),
            "condition": request.form["condition"],
            "warranty_months": int(request.form["warranty_months"])
        }

        if not 0 <= data["age_years"] <= 15:
            raise ValueError("Phone age must be between 0 and 15 years.")
        if not 1000 <= data["original_price"] <= 1000000:
            raise ValueError("Original price must be between Rs. 1,000 and Rs. 10,00,000.")
        if not 1 <= data["ram_gb"] <= 64:
            raise ValueError("RAM must be between 1 and 64 GB.")
        if not 8 <= data["storage_gb"] <= 2048:
            raise ValueError("Storage must be between 8 and 2048 GB.")
        if not 1 <= data["battery_health"] <= 100:
            raise ValueError("Battery health must be between 1 and 100%.")
        if not 0 <= data["warranty_months"] <= 60:
            raise ValueError("Warranty must be between 0 and 60 months.")

        input_df = pd.DataFrame([data])
        estimated_price = max(0, round(float(model.predict(input_df)[0] / 100) * 100))
        resale_percent = estimated_price / data["original_price"] * 100

        return render_template(
            "result.html",
            price=estimated_price,
            percentage=round(resale_percent, 1),
            original_price=data["original_price"],
            brand=data["brand"],
            condition=data["condition"]
        )
    except (ValueError, KeyError) as error:
        return render_template("index.html", error=str(error)), 400
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
