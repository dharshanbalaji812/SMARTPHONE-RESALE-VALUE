"""Train and save the smartphone resale price regression model."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("smartphone_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["brand", "age_years", "original_price", "ram_gb", "storage_gb",
            "battery_health", "condition", "warranty_months"]
TARGET = "resale_price"

if not DATA_PATH.exists():
    raise FileNotFoundError("smartphone_data.csv not found. Run: python generate_dataset.py")

data = pd.read_csv(DATA_PATH)
X = data[FEATURES]
y = data[TARGET]

categorical = ["brand", "condition"]
numeric = [c for c in FEATURES if c not in categorical]
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numeric", "passthrough", numeric)
])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(f"Mean Absolute Error: Rs. {mean_absolute_error(y_test, predictions):,.0f}")
print(f"R-squared: {r2_score(y_test, predictions):.3f}")

joblib.dump(pipeline, MODEL_DIR / "resale_model.pkl")
print("Model saved to models/resale_model.pkl")
