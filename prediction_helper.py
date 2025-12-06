# prediction_helper.py
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb  # noqa: F401  # needed so joblib can load the model

# Paths (same structure as in the notebook)
DATA_DIR = Path(".")
OUT_DIR = Path("./outputs")

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
FEATURES_FILE = DATA_DIR / "features.csv"
STORES_FILE = DATA_DIR / "stores.csv"

MODEL_FILE = OUT_DIR / "final_lightgbm_model.joblib"

TARGET_COL = "Weekly_Sales"


def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar/time-based features from the Date column."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Day"] = df["Date"].dt.day
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols=("Store", "Dept"),
    lags=(7, 14, 28),
) -> pd.DataFrame:
    """
    Add lag features of Weekly_Sales for the given lags (in days).
    We group by (Store, Dept) and then shift Weekly_Sales within each group.
    """
    df = df.sort_values(list(group_cols) + ["Date"]).copy()
    for lag in lags:
        col_name = f"lag_{lag}"
        df[col_name] = (
            df.groupby(list(group_cols))[TARGET_COL]
            .shift(lag)
        )
    return df


def add_rolling_features(
    df: pd.DataFrame,
    group_cols=("Store", "Dept"),
    windows=(7, 14, 28),
) -> pd.DataFrame:
    """
    Add rolling mean features (shifted by 1 to avoid leakage).
    Rolling window is applied on Weekly_Sales within each (Store, Dept).
    """
    df = df.sort_values(list(group_cols) + ["Date"]).copy()
    for win in windows:
        col_name = f"roll_mean_{win}"
        df[col_name] = (
            df.groupby(list(group_cols))[TARGET_COL]
            .shift(1)  # only use strictly past data
            .rolling(window=win)
            .mean()
        )
    return df


def add_hist_stats(
    df: pd.DataFrame,
    group_cols=("Store", "Dept"),
) -> pd.DataFrame:
    """
    Add historical statistics (mean, median, std) per group.
    Uses ONLY rows where Weekly_Sales is known (train part).
    """
    df = df.copy()
    has_target = df[TARGET_COL].notna()

    agg = (
        df.loc[has_target]
        .groupby(list(group_cols))[TARGET_COL]
        .agg(["mean", "median", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "hist_mean",
                "median": "hist_median",
                "std": "hist_std",
            }
        )
    )
    df = df.merge(agg, on=list(group_cols), how="left")
    return df


def load_raw_data(
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four CSVs with the same options as the notebook."""
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["Date"])
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_dir / "stores.csv")

    return train, test, features, stores


def build_feature_data(
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.Index, Dict[str, pd.CategoricalDtype]]:
    """
    Rebuild the feature-engineered data exactly like the notebook.

    Returns
    -------
    test_fe_sorted : pd.DataFrame
        Test set with all engineered features, sorted by Date.
    feature_cols : pd.Index
        List of feature column names used for training.
    cat_dtypes : Dict[str, pd.CategoricalDtype]
        Mapping from feature name -> categorical dtype fitted on train.
    """
    train, test, features, stores = load_raw_data(data_dir)

    # Clean train (same as notebook)
    train = train.copy()
    neg_mask = train[TARGET_COL] < 0
    if neg_mask.any():
        train = train.loc[~neg_mask].copy()

    train = train.drop_duplicates(subset=["Store", "Dept", "Date"])

    # Merge with features & stores
    train_full = (
        train.merge(features, on=["Store", "Date"], how="left")
        .merge(stores, on="Store", how="left")
    )
    test_full = (
        test.merge(features, on=["Store", "Date"], how="left")
        .merge(stores, on="Store", how="left")
    )

    # Combine for feature engineering
    train_full["is_train"] = 1
    test_full["is_train"] = 0
    # Placeholder target so lag functions work
    test_full[TARGET_COL] = np.nan

    all_data = pd.concat([train_full, test_full], ignore_index=True)

    # Feature engineering
    all_data = make_time_features(all_data)
    all_data = add_lag_features(all_data, group_cols=("Store", "Dept"), lags=(7, 14, 28))
    all_data = add_rolling_features(
        all_data,
        group_cols=("Store", "Dept"),
        windows=(7, 14, 28),
    )
    all_data = add_hist_stats(all_data, group_cols=("Store", "Dept"))

    # Split back
    train_fe = all_data[all_data["is_train"] == 1].drop(columns=["is_train"])
    test_fe = all_data[all_data["is_train"] == 0].drop(columns=["is_train"])

    # df_model training subset (needed only to reconstruct feature list + encoders)
    df_model = train_fe.dropna(subset=["lag_7"]).copy()
    df_model = df_model.sort_values("Date").reset_index(drop=True)

    non_feature_cols = [TARGET_COL, "Date", "Store", "Dept"]
    all_cols = df_model.columns.tolist()
    feature_cols = [c for c in all_cols if c not in non_feature_cols]

    # Build categorical dtypes the same way as in the notebook
    X = df_model[feature_cols].copy()
    cat_dtypes: Dict[str, pd.CategoricalDtype] = {}

    for col in feature_cols:
        if X[col].dtype == "object":
            cat_series = X[col].astype("category")
            cat_dtypes[col] = cat_series.dtype

    # Sort test for stable UI
    test_fe_sorted = test_fe.sort_values("Date").reset_index(drop=True)

    return test_fe_sorted, pd.Index(feature_cols), cat_dtypes


def encode_features(
    df: pd.DataFrame,
    feature_cols: pd.Index,
    cat_dtypes: Dict[str, pd.CategoricalDtype],
) -> pd.DataFrame:
    """
    Turn a feature-engineered DataFrame into a numeric matrix for the model.

    Uses the categorical dtypes built from train so that encodings are
    consistent with the model training.
    """
    X = df[list(feature_cols)].copy()

    for col in feature_cols:
        if col in cat_dtypes:
            # Enforce same categories as train
            X[col] = X[col].astype(cat_dtypes[col])
            X[col] = X[col].cat.codes
        else:
            # Ensure numeric dtype for everything else
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # LightGBM can handle NaNs directly
    return X


def load_model(model_path: Path = MODEL_FILE):
    """Load the trained LightGBM model saved by the notebook."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Make sure you ran the training notebook and saved the model first."
        )
    model = joblib.load(model_path)
    return model


def predict_single(
    model,
    row: pd.DataFrame,
    feature_cols: pd.Index,
    cat_dtypes: Dict[str, pd.CategoricalDtype],
) -> float:
    """
    Predict weekly sales for a single (Store, Dept, Date) row from test_fe.

    Parameters
    ----------
    model : LightGBM Booster
    row : pd.DataFrame
        A one-row DataFrame with all engineered features.
    """
    if row.shape[0] != 1:
        raise ValueError("predict_single expects a DataFrame with exactly one row.")
    X_row = encode_features(row, feature_cols, cat_dtypes)
    pred = model.predict(X_row)[0]
    return float(pred)


def add_predictions_to_test(
    model,
    test_fe_sorted: pd.DataFrame,
    feature_cols: pd.Index,
    cat_dtypes: Dict[str, pd.CategoricalDtype],
) -> pd.DataFrame:
    """
    Convenience helper: return a copy of test_fe_sorted with a prediction column.
    """
    X_test = encode_features(test_fe_sorted, feature_cols, cat_dtypes)
    preds = model.predict(X_test)
    out = test_fe_sorted.copy()
    out["Predicted_Weekly_Sales"] = preds
    return out
