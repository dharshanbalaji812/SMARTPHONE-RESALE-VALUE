"""Generate an illustrative sample dataset for the demo project.
Replace this synthetic data with verified real resale listings for meaningful predictions.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
brands = ["Apple", "Samsung", "OnePlus", "Xiaomi", "Vivo", "Oppo", "Realme", "Google"]
conditions = ["Excellent", "Good", "Fair", "Poor"]
brand_factor = {"Apple": 1.12, "Samsung": 1.00, "OnePlus": 0.98, "Xiaomi": 0.88,
                "Vivo": 0.86, "Oppo": 0.87, "Realme": 0.82, "Google": 1.02}
condition_factor = {"Excellent": 1.00, "Good": 0.90, "Fair": 0.76, "Poor": 0.58}
rows = []
for _ in range(700):
    brand = rng.choice(brands)
    age = float(rng.uniform(0.1, 6.0))
    original = int(rng.integers(8000, 150001) // 500 * 500)
    ram = int(rng.choice([2, 3, 4, 6, 8, 12, 16]))
    storage = int(rng.choice([32, 64, 128, 256, 512, 1024]))
    battery = int(rng.integers(55, 101))
    condition = rng.choice(conditions, p=[0.25, 0.42, 0.23, 0.10])
    warranty = int(rng.integers(0, 25))
    age_factor = max(0.12, 1 - 0.16 * age)
    spec_bonus = 1 + min(ram, 16) * 0.008 + min(storage, 1024) / 1024 * 0.06
    battery_factor = 0.72 + battery / 100 * 0.28
    warranty_factor = 1 + min(warranty, 24) * 0.002
    resale = original * age_factor * spec_bonus * battery_factor * condition_factor[condition] * brand_factor[brand] * warranty_factor * rng.normal(1.0, 0.045)
    resale = int(np.clip(resale, original * 0.08, original * 0.95) / 100) * 100
    rows.append([brand, age, original, ram, storage, battery, condition, warranty, resale])
columns = ["brand", "age_years", "original_price", "ram_gb", "storage_gb", "battery_health", "condition", "warranty_months", "resale_price"]
pd.DataFrame(rows, columns=columns).to_csv("smartphone_data.csv", index=False)
print("Created smartphone_data.csv")
