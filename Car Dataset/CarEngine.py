"""Decision Tree model for car prices.

"""
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import train  # for the cleaning functions

df = train.clean(train.read_csv(train.CSV))

X = df.drop(columns="price")
y = np.log1p(df["price"])

cat_cols = ["company", "engine", "fuel"]
num_cols = ["seats", "cc", "hp", "top_speed", "acc_0_100", "torque"]

pre = ColumnTransformer([
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ("num", SimpleImputer(strategy="median"), num_cols),
])

pipe = Pipeline([("pre", pre), ("tree", DecisionTreeRegressor(random_state=42))])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# find the best tree size== (a tree that is too deep memorizes the data)
grid = GridSearchCV(
    pipe,
    {"tree__max_depth": [3, 5, 7, 10, None],
     "tree__min_samples_leaf": [1, 3, 5, 10]},
    cv=5, scoring="r2", n_jobs=-1,
)
grid.fit(X_tr, y_tr)
model = grid.best_estimator_
print("Best settings:", grid.best_params_)

pred = model.predict(X_te)
print(f"R2 (log price): {r2_score(y_te, pred):.3f}")
print(f"MAE (USD):      {mean_absolute_error(np.expm1(y_te), np.expm1(pred)):,.0f}")

# show the top of the tree in readable form
names = model.named_steps["pre"].get_feature_names_out()
print("\nTop of the tree:")
print(export_text(model.named_steps["tree"], feature_names=list(names), max_depth=3))

# predict your own car
car = pd.DataFrame([{
    "company": "BMW", "engine": "V6", "fuel": "petrol",
    "seats": 5, "cc": 2998, "hp": 382, "top_speed": 250,
    "acc_0_100": 4.4, "torque": 500,
}])
print(f"Predicted price: ${np.expm1(model.predict(car)[0]):,.0f}")