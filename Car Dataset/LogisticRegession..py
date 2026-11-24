"""Logistic Regression - is a car EXPENSIVE (above median price)?"""
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

#  classify: expensive (1) or cheap (0) 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

y = (df["price"] > df["price"].median()).astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = make(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
p = model.predict(X_te)
print("Logistic Regression")
print(f"Accuracy: {accuracy_score(y_te, p):.3f}")
print(classification_report(y_te, p, target_names=["cheap", "expensive"]))
prob = model.predict_proba(my_car)[0, 1]
print(f"my_car: {'EXPENSIVE' if prob > 0.5 else 'CHEAP'} (probability expensive = {prob:.2f})")