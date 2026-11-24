"""KNN - car price from engine data. """
import os, re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(os.path.abspath(__file__)))
CSV = "Cars_Datasets_2025.csv"         


def to_num(v):
    """'$12,000-$15,000' -> 13500 ; '70-85 hp' -> 77.5 ; '3,990 cc' -> 3990"""
    if pd.isna(v):
        return np.nan
    nums = re.findall(r"\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(np.mean([float(n) for n in nums[:2]])) if nums else np.nan


try:
    raw = pd.read_csv(CSV)
except UnicodeDecodeError:
    raw = pd.read_csv(CSV, encoding="latin-1")
raw.columns = [c.strip() for c in raw.columns]

df = pd.DataFrame({
    "engine": raw["Engines"].astype(str).str.strip().str.upper(),
    "fuel": raw["Fuel Types"].astype(str).str.strip().str.lower(),
    "cc": raw["CC/Battery Capacity"].map(to_num),
    "hp": raw["HorsePower"].map(to_num),
    "torque": raw["Torque"].map(to_num),
    "price": raw["Cars Prices"].map(to_num),
})
df = df.dropna(subset=["price"])
df = df[df["price"] > 0]

# ENGINE-ONLY features
cat_cols = ["engine", "fuel"]
num_cols = ["cc", "hp", "torque"]
X = df[cat_cols + num_cols]


def make(model):
    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
    ])
    return Pipeline([("pre", pre), ("m", model)])


# example car to predict
my_car = pd.DataFrame([{"engine": "V6", "fuel": "petrol", "cc": 2998, "hp": 382, "torque": 500}])

#predict PRICE (regression) 
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
y = np.log1p(df["price"])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = make(KNeighborsRegressor(n_neighbors=5, weights='distance')).fit(X_tr, y_tr)
p = model.predict(X_te)
print("KNN (k=5)")
print(f"R2 (log price): {r2_score(y_te, p):.3f}")
print(f"MAE (USD):      {mean_absolute_error(np.expm1(y_te), np.expm1(p)):,.0f}")
print(f"Predicted price for my_car: ${np.expm1(model.predict(my_car)[0]):,.0f}")