import pandas as pd
import streamlit as st
from prediction_helper import (
    build_feature_data,
    load_model,
    predict_single,
    add_predictions_to_test,
)

# ---------- Page Config ----------
st.set_page_config(
    page_title="Walmart Weekly Sales Forecasting",
    layout="wide",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-container {
        padding: 20px;
    }
    .metric-card {
        padding: 20px;
        border-radius: 12px;
        background: #f5f5f5;
        border: 1px solid #e3e3e3;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
    }
    h1 {
        font-size: 38px !important;
        font-weight: 700 !important;
    }
    .divider {
        border-top: 1px solid #ddd;
        margin: 25px 0;
    }
    .expander-header {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)



# ---------- Load once ----------
@st.cache_resource(show_spinner=True)
def load_app_objects():
    model = load_model()
    test_fe_sorted, feature_cols, cat_dtypes = build_feature_data()
    test_with_preds = add_predictions_to_test(
        model,
        test_fe_sorted=test_fe_sorted,
        feature_cols=feature_cols,
        cat_dtypes=cat_dtypes,
    )
    return model, test_fe_sorted, feature_cols, cat_dtypes, test_with_preds



# ---------- Main UI ----------
def main():
    st.title("📊 Walmart Weekly Sales Forecasting App")
    st.subheader("Interactive forecasting using trained LightGBM model")

    with st.spinner("Loading model and preparing feature engine..."):
        model, test_fe_sorted, feature_cols, cat_dtypes, test_with_preds = load_app_objects()

    st.sidebar.header("🛒 Select Inputs")
    stores = sorted(test_fe_sorted["Store"].unique().tolist())
    store = st.sidebar.selectbox("Store", stores)

    depts = sorted(test_fe_sorted[test_fe_sorted["Store"] == store]["Dept"].unique().tolist())
    dept = st.sidebar.selectbox("Department", depts)

    subset = test_fe_sorted[(test_fe_sorted["Store"] == store) & (test_fe_sorted["Dept"] == dept)]
    date_options = subset["Date"].dt.strftime("%Y-%m-%d").tolist()
    date_str = st.sidebar.selectbox("Week (Date)", sorted(date_options))
    date = pd.to_datetime(date_str)

    row = subset[subset["Date"] == date]
    if row.empty:
        st.error("❌ No matching data found.")
        return

    # ---------- Prediction Button ----------
    if st.sidebar.button("Predict Weekly Sales"):
        pred = predict_single(
            model=model,
            row=row,
            feature_cols=feature_cols,
            cat_dtypes=cat_dtypes,
        )

        # Metric card display
        st.markdown("### 📌 Forecast Result")
        col_pred, col_info = st.columns([1.2, 1])
        with col_pred:
            st.markdown(f"""
            <div class="metric-card">
                Predicted Weekly Sales<br>
                <span style="font-size:32px;color:#008000;">${pred:,.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_info:
            st.write("**Store:**", store)
            st.write("**Dept:**", dept)
            st.write("**Date:**", date_str)

        with st.expander("🧾 View Feature Row Used for Prediction", expanded=False):
            st.dataframe(row.reset_index(drop=True))


    # ---------- Divider ----------
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ---------- Preview Predictions ----------
    st.header("🗂 Full Forecast Table (Test Set)")
    n_rows = st.slider("Rows to preview", 20, 200, 60, 10)
    st.dataframe(
        test_with_preds[["Store", "Dept", "Date", "Predicted_Weekly_Sales"]]
        .head(n_rows)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    main()
