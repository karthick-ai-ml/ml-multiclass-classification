"""
Streamlit Application — Multi-Class Prediction of Obesity Risk
================================================================
Interactive dashboard for exploring trained obesity-prediction models,
running inference on uploaded CSV or manual feature input, and reviewing
training / dataset analytics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DATA = BASE_DIR / "artifacts" / "data"
ARTIFACTS_REPORTS = BASE_DIR / "artifacts" / "reports"
DATASET_DIR = BASE_DIR / "dataset"
TEST_CSV = ARTIFACTS_DATA / "kaggle_obesity_prediction_test.csv"

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Obesity Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_json(path: str):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)


@st.cache_data
def read_html_report(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────────────────────────────
# Load metadata
# ──────────────────────────────────────────────────────────────────────
model_meta = load_json(str(MODELS_DIR / "kaggle_obesity_prediction_model_trained.json"))
eval_results = load_json(str(ARTIFACTS_DATA / "kaggle_obesity_prediction_model_evaluation_results.json"))
eda_analysis = load_json(str(ARTIFACTS_DATA / "kaggle_obesity_prediction_training_eda_analysis.json"))
preprocess_analysis = load_json(str(ARTIFACTS_DATA / "kaggle_obesity_prediction_preprocessing_analysis.json"))
viz_analysis = load_json(str(ARTIFACTS_DATA / "kaggle_obesity_prediction_visualization_analysis.json"))

FEATURE_NAMES: list[str] = model_meta["dataset"]["feature_names"]  # 22 encoded features
CLASS_NAMES: list[str] = model_meta["dataset"]["class_names"]
LABEL_TO_CLASS: dict = model_meta["dataset"]["label_to_class"]
BEST_MODEL_NAME: str = model_meta["best_model"]["name"]

# Scaling params
SCALING: dict = preprocess_analysis.get("scaling_params", preprocess_analysis["preprocessing_stats"].get("scaling", {}))

# Build model registry: name -> file
MODEL_REGISTRY: dict[str, str] = {}
for name, info in model_meta["all_models"].items():
    MODEL_REGISTRY[name] = str(MODELS_DIR / info["model_file"])


# ──────────────────────────────────────────────────────────────────────
# Helper: preprocess raw row(s) into encoded + scaled feature array
# ──────────────────────────────────────────────────────────────────────

# Raw feature definitions (pre-encoding)
RAW_NUMERICAL = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
RAW_BINARY = {
    "Gender": ["Female", "Male"],
    "family_history_with_overweight": ["no", "yes"],
    "FAVC": ["no", "yes"],
    "SMOKE": ["no", "yes"],
    "SCC": ["no", "yes"],
}
RAW_CATEGORICAL = {
    "CAEC": ["Always", "Frequently", "Sometimes", "no"],
    "CALC": ["Frequently", "Sometimes", "no"],
    "MTRANS": ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"],
}

# Pretty labels for display
FEATURE_LABELS = {
    "Age": "Age (years)",
    "Height": "Height (m)",
    "Weight": "Weight (kg)",
    "FCVC": "Vegetable consumption frequency (1-3)",
    "NCP": "Number of main meals (1-4)",
    "CH2O": "Daily water intake (1-3)",
    "FAF": "Physical activity frequency (0-3)",
    "TUE": "Technology usage time (0-2)",
    "Gender": "Gender",
    "family_history_with_overweight": "Family history of overweight",
    "FAVC": "Frequent high-caloric food consumption",
    "SMOKE": "Do you smoke?",
    "SCC": "Calorie consumption monitoring",
    "CAEC": "Consumption of food between meals",
    "CALC": "Alcohol consumption",
    "MTRANS": "Main transportation used",
}

NUMERICAL_RANGES = {
    "Age": (14.0, 61.0, 25.0),
    "Height": (1.45, 1.98, 1.70),
    "Weight": (39.0, 165.0, 80.0),
    "FCVC": (1.0, 3.0, 2.0),
    "NCP": (1.0, 4.0, 3.0),
    "CH2O": (1.0, 3.0, 2.0),
    "FAF": (0.0, 3.0, 1.0),
    "TUE": (0.0, 2.0, 0.5),
}

# Class display colours (Plotly-friendly)
CLASS_COLORS = {
    "Insufficient_Weight": "#2ecc71",
    "Normal_Weight": "#27ae60",
    "Overweight_Level_I": "#f39c12",
    "Overweight_Level_II": "#e67e22",
    "Obesity_Type_I": "#e74c3c",
    "Obesity_Type_II": "#c0392b",
    "Obesity_Type_III": "#8e44ad",
}


def encode_and_scale(df_raw: pd.DataFrame) -> np.ndarray:
    """Convert a raw-feature DataFrame into the model's 22-dim scaled array."""
    records = []
    for _, row in df_raw.iterrows():
        vec: dict[str, float] = {}
        # Numerical (keep as-is first)
        for col in RAW_NUMERICAL:
            vec[col] = float(row[col])
        # Binary one-hot
        for col, categories in RAW_BINARY.items():
            hot_col = f"{col}_{categories[1]}"  # _Male, _yes etc.
            vec[hot_col] = 1.0 if str(row[col]) == categories[1] else 0.0
        # Multi-category one-hot (drop-first already handled in training)
        for col, categories in RAW_CATEGORICAL.items():
            encoded_cols = preprocess_analysis["preprocessing_stats"]["encoding"][col]["new_columns"]
            for ec in encoded_cols:
                cat_val = ec.replace(f"{col}_", "")
                vec[ec] = 1.0 if str(row[col]) == cat_val else 0.0
        # Assemble in feature order
        arr = []
        for feat in FEATURE_NAMES:
            val = vec.get(feat, 0.0)
            # Standard-scale using training stats
            if feat in SCALING:
                mean = SCALING[feat]["mean"]
                std = SCALING[feat]["std"]
                if std > 0:
                    val = (val - mean) / std
            arr.append(val)
        records.append(arr)
    return np.array(records)


def predict(model, X: np.ndarray):
    """Return (predicted_labels, probability_matrix)."""
    preds = model.predict(X)
    probas = None
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
    return preds, probas


# ──────────────────────────────────────────────────────────────────────
# Sidebar — model selector
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-health.png", width=64)
    st.title("⚙️ Settings")
    model_names = list(MODEL_REGISTRY.keys())
    default_idx = model_names.index(BEST_MODEL_NAME) if BEST_MODEL_NAME in model_names else 0
    selected_model_name = st.selectbox(
        "Select Model",
        model_names,
        index=default_idx,
        help="Best model (XGBoost Ensemble) is selected by default based on validation performance.",
    )
    if selected_model_name == BEST_MODEL_NAME:
        st.success(f"✅ **Best Model** — {BEST_MODEL_NAME}")
    model = load_model(MODEL_REGISTRY[selected_model_name])

    st.divider()
    st.caption("Multi-Class Prediction of Obesity Risk")
    st.caption("BITS Pilani — ML Assignment 2")

# ──────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────
tab_predict, tab_models, tab_dataset = st.tabs([
    "🔮 Predict",
    "📊 Model Performance",
    "📁 Dataset Information",
])

# ======================================================================
# TAB 1 — Prediction
# ======================================================================
with tab_predict:
    st.header("Obesity Risk Prediction")
    st.markdown(
        "Upload a CSV file **or** manually enter feature values below to predict the obesity risk category."
    )

    # ---- Download test CSV ----
    st.subheader("📥 Download Test Dataset")
    st.markdown(
        "Download the sample test CSV. You can use it to upload and validate predictions."
    )
    if TEST_CSV.exists():
        test_bytes = TEST_CSV.read_bytes()
        st.download_button(
            label="⬇️  Download kaggle_obesity_prediction_test.csv",
            data=test_bytes,
            file_name="kaggle_obesity_prediction_test.csv",
            mime="text/csv",
        )
    else:
        st.warning("Test CSV not found at expected path.")

    st.divider()

    # ---- Upload CSV ----
    st.subheader("📤 Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader(
        "Upload a CSV with raw features (same schema as test.csv)",
        type=["csv"],
        help="Columns expected: Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS. Optionally 'id' and 'NObeyesdad'.",
    )

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.write(f"**Uploaded rows:** {len(df_upload)}")
        st.dataframe(df_upload.head(10), use_container_width=True)

        required_cols = set(RAW_NUMERICAL) | set(RAW_BINARY.keys()) | set(RAW_CATEGORICAL.keys())
        missing = required_cols - set(df_upload.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            with st.spinner("Running predictions…"):
                X_enc = encode_and_scale(df_upload)
                labels, probas = predict(model, X_enc)
                pred_classes = [LABEL_TO_CLASS.get(str(int(l)), str(l)) for l in labels]
                df_upload["Predicted_Class"] = pred_classes

                if probas is not None:
                    max_probs = probas.max(axis=1)
                    df_upload["Confidence (%)"] = (max_probs * 100).round(2)

                # Compare with actual if available
                has_actual = "NObeyesdad" in df_upload.columns
                if has_actual:
                    df_upload["Actual_Class"] = df_upload["NObeyesdad"]
                    df_upload["Correct"] = df_upload["Predicted_Class"] == df_upload["Actual_Class"]
                    accuracy = df_upload["Correct"].mean()
                    st.metric("Batch Accuracy", f"{accuracy:.2%}")

            display_cols = ["id"] if "id" in df_upload.columns else []
            if has_actual:
                display_cols += ["Actual_Class"]
            display_cols += ["Predicted_Class"]
            if "Confidence (%)" in df_upload.columns:
                display_cols += ["Confidence (%)"]
            if has_actual:
                display_cols += ["Correct"]

            st.dataframe(
                df_upload[display_cols + [c for c in df_upload.columns if c not in display_cols]].head(50),
                use_container_width=True,
                height=400,
            )

            # Download predictions
            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Download Predictions CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
            )

    st.divider()

    # ---- OR: Manual Input ----
    st.subheader("✏️ Manual Feature Input")
    st.markdown("Fill in the feature values and click **Diagnose** to get a prediction.")

    col_left, col_right = st.columns(2)

    input_values: dict = {}

    with col_left:
        st.markdown("**Numerical Features**")
        for feat in RAW_NUMERICAL:
            lo, hi, default = NUMERICAL_RANGES[feat]
            step = 0.01 if feat in ("Height",) else 0.1 if feat in ("FCVC", "NCP", "CH2O", "FAF", "TUE") else 1.0
            input_values[feat] = st.slider(
                FEATURE_LABELS.get(feat, feat),
                min_value=lo,
                max_value=hi,
                value=default,
                step=step,
                key=f"manual_{feat}",
            )

    with col_right:
        st.markdown("**Categorical / Binary Features**")
        for feat, cats in RAW_BINARY.items():
            input_values[feat] = st.selectbox(
                FEATURE_LABELS.get(feat, feat),
                cats,
                index=1 if feat != "SMOKE" else 0,
                key=f"manual_{feat}",
            )
        for feat, cats in RAW_CATEGORICAL.items():
            default_idx = cats.index("Sometimes") if "Sometimes" in cats else 0
            input_values[feat] = st.selectbox(
                FEATURE_LABELS.get(feat, feat),
                cats,
                index=default_idx,
                key=f"manual_{feat}",
            )

    if st.button("🩺 Diagnose", type="primary", use_container_width=True):
        df_manual = pd.DataFrame([input_values])
        X_manual = encode_and_scale(df_manual)
        label, probas = predict(model, X_manual)
        pred_class = LABEL_TO_CLASS.get(str(int(label[0])), str(label[0]))

        st.divider()
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            colour = CLASS_COLORS.get(pred_class, "#3498db")
            st.markdown(
                f"""
                <div style="background:{colour}22; border-left:5px solid {colour};
                            padding:20px; border-radius:8px; margin-bottom:16px;">
                    <h2 style="color:{colour}; margin:0;">🎯 {pred_class.replace('_', ' ')}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if probas is not None:
                confidence = probas[0].max() * 100
                st.metric("Confidence", f"{confidence:.1f}%")

        with col_res2:
            if probas is not None:
                prob_df = pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Probability (%)": (probas[0] * 100).round(2),
                })
                prob_df = prob_df.sort_values("Probability (%)", ascending=True)
                fig = px.bar(
                    prob_df,
                    x="Probability (%)",
                    y="Class",
                    orientation="h",
                    color="Class",
                    color_discrete_map=CLASS_COLORS,
                    title="Prediction Probabilities",
                )
                fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# TAB 2 — Model Performance
# ======================================================================
with tab_models:
    st.header("Model Training Results & Performance")

    # ---- Validation metrics comparison ----
    st.subheader("Validation Metrics Comparison")
    val_df = pd.DataFrame(eval_results["validation_metrics"])
    val_df_display = val_df.copy()
    for c in val_df_display.columns:
        if c != "Model":
            val_df_display[c] = val_df_display[c].apply(lambda x: f"{x:.4f}")
    st.dataframe(val_df_display, use_container_width=True, hide_index=True)

    # ---- Training metrics comparison ----
    st.subheader("Training Metrics Comparison")
    train_df = pd.DataFrame(eval_results["training_metrics"])
    train_df_display = train_df.copy()
    for c in train_df_display.columns:
        if c != "Model":
            train_df_display[c] = train_df_display[c].apply(lambda x: f"{x:.4f}")
    st.dataframe(train_df_display, use_container_width=True, hide_index=True)

    # ---- Model observations ----
    st.subheader("Model Observations")
    obs_df = pd.DataFrame(eval_results["model_observations"])
    st.dataframe(obs_df, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Best model summary ----
    st.subheader(f"🏆 Best Model: {BEST_MODEL_NAME}")
    best_info = model_meta["best_model"]
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("**Validation Metrics**")
        for k, v in best_info["validation_metrics"].items():
            st.metric(k, f"{v:.4f}")
    with col_b2:
        st.markdown("**Combined (Train + Val) Metrics**")
        for k, v in best_info["combined_metrics"].items():
            st.metric(k, f"{v:.4f}")

    st.markdown(f"**Cross-Validation Score:** {best_info['cv_score']:.4f} ± {best_info['cv_std']:.4f}")

    st.divider()

    # ---- Interactive charts ----
    st.subheader("Metric Comparisons")

    metric_choice = st.selectbox(
        "Select metric to compare",
        ["Accuracy", "AUC_Score", "Precision", "Recall", "F1_Score", "MCC_Score"],
        index=0,
    )

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="Training",
        x=train_df["Model"],
        y=train_df[metric_choice],
        marker_color="#3498db",
    ))
    fig_cmp.add_trace(go.Bar(
        name="Validation",
        x=val_df["Model"],
        y=val_df[metric_choice],
        marker_color="#e74c3c",
    ))
    fig_cmp.update_layout(
        barmode="group",
        title=f"{metric_choice} — Training vs Validation",
        yaxis_title=metric_choice,
        height=420,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ---- Radar chart of validation metrics ----
    st.subheader("Validation Metrics Radar")
    radar_model = st.selectbox("Select model for radar", val_df["Model"].tolist(), key="radar_model")
    radar_row = val_df[val_df["Model"] == radar_model].iloc[0]
    metrics_for_radar = ["Accuracy", "AUC_Score", "Precision", "Recall", "F1_Score", "MCC_Score"]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[radar_row[m] for m in metrics_for_radar] + [radar_row[metrics_for_radar[0]]],
        theta=metrics_for_radar + [metrics_for_radar[0]],
        fill="toself",
        name=radar_model,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Validation Metrics — {radar_model}",
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ---- Embedded HTML reports ----
    st.subheader("Detailed Reports (from training pipeline)")
    report_files = {
        "Validation Metrics Heatmap": "kaggle_obesity_prediction_validation_metrics_heatmap.html",
        "Training Metrics Heatmap": "kaggle_obesity_prediction_training_metrics_heatmap.html",
        "Model Comparison (Validation)": "kaggle_obesity_prediction_model_comparison_validation_metrics.html",
        "Confusion Matrices": "kaggle_obesity_prediction_confusion_matrices.html",
        "AUC-ROC Curves": "kaggle_obesity_prediction_auc_roc_curves.html",
        "Model Observations": "kaggle_obesity_prediction_model_observations.html",
    }

    selected_report = st.selectbox("Choose report", list(report_files.keys()))
    report_path = ARTIFACTS_REPORTS / report_files[selected_report]
    if report_path.exists():
        html_content = read_html_report(str(report_path))
        st.components.v1.html(html_content, height=700, scrolling=True)
    else:
        st.warning(f"Report file not found: {report_path.name}")


# ======================================================================
# TAB 3 — Dataset Information
# ======================================================================
with tab_dataset:
    st.header("Dataset Information")

    # Summary cards
    summary = eda_analysis.get("overall_summary", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Samples", f"{summary.get('training_samples', 'N/A'):,}")
    c2.metric("Total Features", summary.get("total_features", "N/A"))
    c3.metric("Target Classes", summary.get("num_classes", "N/A"))
    c4.metric("Missing Values", summary.get("has_missing_values", "N/A"))

    st.divider()

    # ---- Target distribution ----
    st.subheader("Target Class Distribution")
    target = eda_analysis["training_analysis"]["target_analysis"]
    dist = target["class_distribution"]
    dist_df = pd.DataFrame([
        {"Class": k, "Count": v["count"], "Percentage": round(v["percentage"], 2)}
        for k, v in dist.items()
    ])
    fig_target = px.bar(
        dist_df,
        x="Class",
        y="Count",
        color="Class",
        color_discrete_map=CLASS_COLORS,
        text="Percentage",
        title="NObeyesdad — Target Distribution",
    )
    fig_target.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_target.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig_target, use_container_width=True)

    st.divider()

    # ---- Feature categorisation ----
    st.subheader("Feature Categorization")
    feat_cat = eda_analysis.get("feature_categorization", {})
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("**Numerical**")
        for f in feat_cat.get("numerical", []):
            if f != "id":
                st.markdown(f"- `{f}`")
    with col_f2:
        st.markdown("**Categorical**")
        for f in feat_cat.get("categorical", []):
            st.markdown(f"- `{f}`")
    with col_f3:
        st.markdown("**Binary**")
        for f in feat_cat.get("binary", []):
            st.markdown(f"- `{f}`")

    st.divider()

    # ---- Numerical statistics ----
    st.subheader("Numerical Feature Statistics")
    num_stats = eda_analysis["training_analysis"].get("numerical_analysis", {})
    if num_stats:
        rows = []
        for feat, stats in num_stats.items():
            if feat == "id":
                continue
            rows.append({
                "Feature": feat,
                "Mean": round(stats["mean"], 3),
                "Std": round(stats["std"], 3),
                "Min": round(stats["min"], 2),
                "Q1": round(stats["q1"], 2),
                "Median": round(stats["median"], 2),
                "Q3": round(stats["q3"], 2),
                "Max": round(stats["max"], 2),
                "Skewness": round(stats["skewness"], 3),
                "Kurtosis": round(stats["kurtosis"], 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ---- Categorical analysis ----
    st.subheader("Categorical Feature Distribution")
    cat_analysis = eda_analysis["training_analysis"].get("categorical_analysis", {})
    if cat_analysis:
        cat_feat_pick = st.selectbox("Select feature", list(cat_analysis.keys()))
        cat_info = cat_analysis[cat_feat_pick]
        cat_df = pd.DataFrame([
            {"Category": k, "Count": v}
            for k, v in cat_info["top_values"].items()
        ])
        fig_cat = px.pie(
            cat_df,
            names="Category",
            values="Count",
            title=f"{cat_feat_pick} Distribution",
            hole=0.35,
        )
        fig_cat.update_layout(height=380)
        st.plotly_chart(fig_cat, use_container_width=True)

    st.divider()

    # ---- Data quality ----
    st.subheader("Data Quality")
    dq = viz_analysis.get("data_quality", {})
    q1, q2, q3 = st.columns(3)
    q1.metric("Missing Values", dq.get("missing_values", "N/A"))
    q2.metric("Duplicate Rows", dq.get("duplicate_rows", "N/A"))
    q3.metric("Completeness", f"{dq.get('completeness_rate', 'N/A')}%")

    st.divider()

    # ---- Sample data ----
    st.subheader("Sample Data (Head)")
    sample_head = ARTIFACTS_DATA / "kaggle_obesity_prediction_sample_head_10.csv"
    if sample_head.exists():
        st.dataframe(load_csv(str(sample_head)), use_container_width=True, hide_index=True)

    st.divider()

    # ---- Preprocessing information ----
    st.subheader("Preprocessing Summary")
    pp = preprocess_analysis
    pp1, pp2, pp3 = st.columns(3)
    pp1.metric("Train Shape", f"{pp['train_shape'][0]} × {pp['train_shape'][1]}")
    pp2.metric("Validation Shape", f"{pp['validation_shape'][0]} × {pp['validation_shape'][1]}")
    pp3.metric("Test Shape", f"{pp['test_shape'][0]} × {pp['test_shape'][1]}")

    st.markdown("**Encoding Summary**")
    enc_rows = []
    for feat, info in pp["preprocessing_stats"]["encoding"].items():
        enc_rows.append({
            "Feature": feat,
            "Encoding": info["encoding_type"],
            "New Columns": ", ".join(info["new_columns"]),
            "Num Categories": info["num_categories"],
        })
    st.dataframe(pd.DataFrame(enc_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ---- Embedded HTML reports (dataset-related) ----
    st.subheader("Visual Reports")
    ds_report_files = {
        "Dataset Overview": "kaggle_obesity_prediction_dataset_overview.html",
        "Feature Summary": "kaggle_obesity_prediction_feature_summary.html",
        "Target Distribution": "kaggle_obesity_prediction_target_distribution.html",
        "Numerical Distributions": "kaggle_obesity_prediction_numerical_distributions.html",
        "Categorical Distributions": "kaggle_obesity_prediction_categorical_distributions.html",
        "Correlation Matrix": "kaggle_obesity_prediction_correlation_matrix.html",
    }

    selected_ds_report = st.selectbox("Choose report", list(ds_report_files.keys()), key="ds_report")
    ds_path = ARTIFACTS_REPORTS / ds_report_files[selected_ds_report]
    if ds_path.exists():
        html_ds = read_html_report(str(ds_path))
        st.components.v1.html(html_ds, height=700, scrolling=True)
    else:
        st.warning(f"Report file not found: {ds_path.name}")

# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown(
    """
    <div style="text-align:center; color: #888; font-size: 0.8em;">
        Multi-Class Prediction of Obesity Risk<br>
        BITS Pilani — ML Assignment 2<br>
        © 2026
    </div>
    """,
    unsafe_allow_html=True,
)
