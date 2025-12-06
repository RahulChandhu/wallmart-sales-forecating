# predictionhelper.py

import numpy as np
import pandas as pd
from pathlib import Path
import joblib


# =============================
# DATA & MODEL PATHS
# =============================
DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
STORES_PATH = DATA_DIR / "stores.csv"

# MODEL IN CURRENT FOLDER (NO OUTPUTS DIRECTORY)
MODEL_PATH = Path("final_lightgbm_model.joblib")


# =============================
# LOAD MODEL
# =============================
def load_trained_model():
    return joblib.load(MODEL_PATH)


# =============================
# FEATURE ENGINEERING
# =============================
def make_time_features(df):
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Day"] = df["Date"].dt.day
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    return df


def add_lag_features(df, group_cols=("Store", "Dept"), lags=(7, 14, 28)):
    df = df.sort_values(list(group_cols) + ["Date"]).copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(list(group_cols))["Weekly_Sales"].shift(lag)
    return df


def add_rolling_features(df, group_cols=("Store", "Dept"), windows=(7, 14, 28)):
    df = df.sort_values(list(group_cols) + ["Date"]).copy()
    for w in windows:
        df[f"roll_mean_{w}"] = (
            df.groupby(list(group_cols))["Weekly_Sales"].shift(1).rolling(window=w).mean()
        )
    return df


def add_hist_stats(df, group_cols=("Store", "Dept")):
    df = df.copy()
    valid = df["Weekly_Sales"].notna()

    agg = (
        df.loc[valid]
        .groupby(list(group_cols))["Weekly_Sales"]
        .agg(["mean", "median", "std"])
        .reset_index()
        .rename(columns={"mean": "hist_mean", "median": "hist_median", "std": "hist_std"})
    )
    return df.merge(agg, on=list(group_cols), how="left")


# =============================
# RAW DATA LOADER
# =============================
def load_raw_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test = pd.read_csv(TEST_PATH, parse_dates=["Date"])
    features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    stores = pd.read_csv(STORES_PATH)
    return train, test, features, stores


# =============================
# BUILD FULL FEATURE SET
# =============================
def prepare_features():
    train, test, features, stores = load_raw_data()

    # Clean Train
    train = train[train["Weekly_Sales"] >= 0].copy()
    train = train.drop_duplicates(subset=["Store", "Dept", "Date"]).copy()

    # Merge
    train_full = (
        train.merge(features, on=["Store", "Date"], how="left")
        .merge(stores, on="Store", how="left")
    )
    test_full = (
        test.merge(features, on=["Store", "Date"], how="left")
        .merge(stores, on="Store", how="left")
    )

    train_full["is_train"] = 1
    test_full["is_train"] = 0
    test_full["Weekly_Sales"] = np.nan

    all_data = pd.concat([train_full, test_full], ignore_index=True)

    # Feature Engineering
    all_data = make_time_features(all_data)
    all_data = add_lag_features(all_data)
    all_data = add_rolling_features(all_data)
    all_data = add_hist_stats(all_data)

    train_fe = all_data[all_data["is_train"] == 1].drop(columns=["is_train"])
    test_fe = all_data[all_data["is_train"] == 0].drop(columns=["is_train"])

    df_model = train_fe.dropna(subset=["lag_7"]).copy()

    TARGET = "Weekly_Sales"
    non_feat = [TARGET, "Date", "Store", "Dept"]
    feature_cols = [c for c in df_model.columns if c not in non_feat]

    return train_fe, test_fe, feature_cols


# =============================
# ENCODING
# =============================
def encode(X):
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes
    return X


# =============================
# PREDICT TEST (SUBMISSION)
# =============================
def build_submission_df():
    _, test_fe, feat_cols = prepare_features()
    model = load_trained_model()

    test_sorted = test_fe.sort_values("Date").reset_index(drop=True)
    X_test = encode(test_sorted[feat_cols])

    preds = model.predict(X_test)

    out = test_sorted[["Store", "Dept", "Date"]].copy()
    out["Weekly_Sales"] = preds
    return out


# =============================
# PREDICT TRAIN (FOR METRICS)
# =============================
def build_train_predictions():
    train_fe, _, feat_cols = prepare_features()
    df = train_fe.dropna(subset=["lag_7"]).copy()

    X = encode(df[feat_cols])
    y_true = df["Weekly_Sales"].values

    model = load_trained_model()
    preds = model.predict(X)

    out = df[["Store", "Dept", "Date"]].copy()
    out["Weekly_Sales"] = y_true
    out["Predicted"] = preds
    return out


# =============================
# HELPERS FOR STREAMLIT
# =============================
def get_store_options(sub):
    return sorted(sub["Store"].unique())


def get_dept_options(sub, store):
    return sorted(sub[sub["Store"] == store]["Dept"].unique())


def filter_predictions(sub, store, dept):
    return sub[(sub["Store"] == store) & (sub["Dept"] == dept)].sort_values("Date")

