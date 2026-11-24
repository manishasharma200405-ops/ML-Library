"""K-Means - group cars into clusters by engine data (no labels needed)."""
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


my_car = pd.DataFrame([{"engine": "V6", "fuel": "petrol", "cc": 2998, "hp": 382, "torque": 500}])

# unsupervised: cluster cars by cc, hp, torque 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

feat = df[num_cols].copy()
feat = feat.fillna(feat.median())
Z = StandardScaler().fit_transform(feat)

print("Choosing k (higher silhouette = better separated clusters):")
for k in range(2, 8):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
    print(f"  k={k}  silhouette={silhouette_score(Z, labels):.3f}")

K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Z)
df["cluster"] = km.labels_

print(f"\nCluster summary (k={K}):")
summary = df.groupby("cluster").agg(
    cars=("price", "size"), avg_cc=("cc", "mean"), avg_hp=("hp", "mean"),
    avg_torque=("torque", "mean"), avg_price=("price", "mean"),
).round(0)
print(summary.to_string())

sc = StandardScaler().fit(feat)
my_z = sc.transform(my_car[num_cols])
print(f"\nmy_car belongs to cluster {km.predict(my_z)[0]}")