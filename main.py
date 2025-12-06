# ProperMain.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from prediction_helper import (
    build_submission_df,
    build_train_predictions,
    get_store_options,
    get_dept_options,
    filter_predictions,
)


# ==============================
# FIXED DATE HANDLER
# ==============================
def parse_date_input(x):
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return pd.to_datetime(x[0]), pd.to_datetime(x[1])
    else:
        d = pd.to_datetime(x)
        return d, d


# ==============================
# LOAD DATA (cached)
# ==============================
@st.cache_data(show_spinner=True)
def load_all():
    sub = build_submission_df()
    train = build_train_predictions()
    return sub, train


submission, train_pred = load_all()


# ==============================
# UI HEADER
# ==============================
st.set_page_config(page_title="Walmart Sales", layout="wide")
st.markdown("<h2 style='text-align:center;color:#1E3A8A;'>Walmart Weekly Sales Forecasting</h2>", unsafe_allow_html=True)


# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Filters")

view_mode = st.sidebar.radio(
    "Mode",
    ["Single Store & Department", "Compare Stores", "Compare Departments"]
)

# Date range
min_date = min(submission["Date"].min(), train_pred["Date"].min()).date()
max_date = max(submission["Date"].max(), train_pred["Date"].max()).date()

date_input = st.sidebar.date_input("Date Range", value=[min_date, max_date])
start_date, end_date = parse_date_input(date_input)

show_table = st.sidebar.checkbox("Show Table", value=False)

predict_btn = st.sidebar.button("Run Forecast")


# ==============================
# GLOBAL PERFORMANCE
# ==============================
def rmse_mae(y_true, y_pred):
    if len(y_true) == 0: return None, None
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return rmse, mae


global_rmse, global_mae = rmse_mae(train_pred["Weekly_Sales"], train_pred["Predicted"])
st.metric("Global RMSE", f"{global_rmse:,.2f}")
st.metric("Global MAE", f"{global_mae:,.2f}")


# ==============================
# HELPER
# ==============================
def filter_date(df):
    return df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]


# ==============================
# MAIN LOGIC
# ==============================
if not predict_btn:
    st.info("Set filters and click **Run Forecast**")
    st.stop()

# ---------------- SINGLE MODE ----------------
if view_mode == "Single Store & Department":

    store_list = get_store_options(submission)
    selected_store = st.sidebar.selectbox("Select Store", store_list)

    dept_list = get_dept_options(submission, selected_store)
    selected_dept = st.sidebar.selectbox("Select Department", dept_list)

    forecast = filter_predictions(submission, selected_store, selected_dept)
    forecast = filter_date(forecast)

    history = train_pred[(train_pred["Store"] == selected_store) & (train_pred["Dept"] == selected_dept)]
    history = filter_date(history)

    st.subheader(f"Forecast — Store {selected_store}, Dept {selected_dept}")

    # Build chart dataframe
    parts = []
    if not history.empty:
        parts.append(history.set_index("Date")["Weekly_Sales"].rename("Actual(train)"))
    if not forecast.empty:
        parts.append(forecast.set_index("Date")["Weekly_Sales"].rename("Forecast(test)"))

    chart_df = pd.concat(parts, axis=1)

    st.line_chart(chart_df)

    if show_table:
        st.dataframe(chart_df)

# ---------------- COMPARE STORES ----------------
elif view_mode == "Compare Stores":

    dept_all = sorted(submission["Dept"].unique())
    selected_dept = st.sidebar.selectbox("Department", dept_all)

    store_all = get_store_options(submission)
    default = store_all[:3]
    selected = st.sidebar.multiselect("Stores", store_all, default)

    df = submission[(submission["Dept"] == selected_dept) & (submission["Store"].isin(selected))]
    df = filter_date(df)

    chart = df.pivot(index="Date", columns="Store", values="Weekly_Sales")
    st.line_chart(chart)

    if show_table:
        st.dataframe(df)

# ---------------- COMPARE DEPARTMENTS ----------------
else:
    store_all = get_store_options(submission)
    selected_store = st.sidebar.selectbox("Store", store_all)

    dept = get_dept_options(submission, selected_store)
    default = dept[:3]
    selected = st.sidebar.multiselect("Depts", dept, default)

    df = submission[(submission["Store"] == selected_store) & (submission["Dept"].isin(selected))]
    df = filter_date(df)

    chart = df.pivot(index="Date", columns="Dept", values="Weekly_Sales")
    st.line_chart(chart)

    if show_table:
        st.dataframe(df)

