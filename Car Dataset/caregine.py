"""Car price prediction model.

"""
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CSV = "Cars_Datasets_2025.csv"


def read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def to_num(value):
    """'$12,000-$15,000' -> 13500 ; '70-85 hp' -> 77.5 ; '3,990 cc' -> 3990."""
    if pd.isna(value):
        return np.nan
    nums = re.findall(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not nums:
        return np.nan
    nums = [float(n) for n in nums[:2]]
    return float(np.mean(nums))


def clean(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame()
    out["company"] = df["Company Names"].str.strip().str.upper()
    out["engine"] = df["Engines"].astype(str).str.strip().str.upper()
    out["fuel"] = df["Fuel Types"].astype(str).str.strip().str.lower()
    out["seats"] = df["Seats"].map(to_num)
    out["cc"] = df["CC/Battery Capacity"].map(to_num)
    out["hp"] = df["HorsePower"].map(to_num)
    out["top_speed"] = df["Total Speed"].map(to_num)
    out["acc_0_100"] = df["Performance(0 - 100 )KM/H"].map(to_num)
    out["torque"] = df["Torque"].map(to_num)
    out["price"] = df["Cars Prices"].map(to_num)
    out = out.dropna(subset=["price"])
    out = out[out["price"] > 0]
    return out


def main():
    df = clean(read_csv(CSV))
    print(f"Rows after cleaning: {len(df)}")

    X = df.drop(columns="price")
    y = np.log1p(df["price"])  # prices are heavily skewed, predict log price

    cat_cols = ["company", "engine", "fuel"]
    num_cols = ["seats", "cc", "hp", "top_speed", "acc_0_100", "torque"]

    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ])

    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    print(f"R2 (log price): {r2_score(y_te, pred):.3f}")
    print(f"MAE (USD):      {mean_absolute_error(np.expm1(y_te), np.expm1(pred)):,.0f}")
    print(f"Median abs % error: {np.median(np.abs(np.expm1(pred) / np.expm1(y_te) - 1)) * 100:.1f}%")

    joblib.dump(model, "car_price_model.joblib")
    print("Saved car_price_model.joblib")

    example = pd.DataFrame([{
        "company": "BMW", "engine": "V6", "fuel": "petrol",
        "seats": 5, "cc": 2998, "hp": 382, "top_speed": 250,
        "acc_0_100": 4.4, "torque": 500,
    }])
    print(f"Example prediction: ${np.expm1(model.predict(example)[0]):,.0f}")


if __name__ == "__main__":
    main()