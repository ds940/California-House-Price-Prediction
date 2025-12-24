import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# -----------------------------
# Load Dataset
# -----------------------------
housing = pd.read_csv("housing.csv")

# -----------------------------
# Create Income Category
# -----------------------------
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

# -----------------------------
# Stratified Split (LIKE PIC)
# -----------------------------
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    # save test set WITH label as input.csv
    housing.loc[test_index].drop("income_cat", axis=1).to_csv(
        "input.csv", index=False
    )

    # use train set for training
    housing = housing.loc[train_index].drop("income_cat", axis=1)

# -----------------------------
# Separate Features & Labels
# -----------------------------
housing_labels = housing["median_house_value"].copy()
housing_features = housing.drop("median_house_value", axis=1)

# -----------------------------
# Pipelines
# -----------------------------
num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# ==================================================
# Train or Inference
# ==================================================
if not os.path.exists(MODEL_FILE):

    print("Training Random Forest model...")

    housing_prepared = full_pipeline.fit_transform(housing_features)

    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(housing_prepared, housing_labels)

    rf_preds = rf_reg.predict(housing_prepared)
    rf_rmse = root_mean_squared_error(housing_labels, rf_preds)
    print("Random Forest Training RMSE:", rf_rmse)

    rf_scores = -cross_val_score(
        rf_reg,
        housing_prepared,
        housing_labels,
        scoring="neg_root_mean_squared_error",
        cv=10
    )
    print(pd.Series(rf_scores).describe())

    joblib.dump(rf_reg, MODEL_FILE)
    joblib.dump(full_pipeline, PIPELINE_FILE)

    print("Model is trained. Congrats! 🎉")

else:
    # -----------------------------
    # Inference (LIKE PIC)
    # -----------------------------
    print("Loading model for inference...")

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")

    # drop true label before transform
    input_features = input_data.drop("median_house_value", axis=1)

    transformed_input = pipeline.transform(input_features)

    predictions = model.predict(transformed_input)

    # replace with predicted values
    input_data["median_house_value"] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Inference is complete, results saved to output.csv. Enjoy! 🚀")
