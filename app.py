"""
Streamlit Application — Multi-Class Prediction of Obesity Risk
================================================================
Interactive dashboard for exploring trained obesity-prediction models,
running inference on uploaded CSV or manual feature input, and reviewing
training / dataset analytics.
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

# Suppress XGBoost serialization warning when loading older pickle models
warnings.filterwarnings("ignore", message=".*If you are loading a serialized model.*")
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
ARTIFACTS_IMAGES = BASE_DIR / "artifacts" / "images"
ARTIFACTS_REPORTS = BASE_DIR / "artifacts" / "reports"
DATASET_DIR = BASE_DIR / "dataset"
TEST_CSV = ARTIFACTS_DATA / "kaggle_obesity_prediction_test.csv"
ROBOTIC_PNG = ARTIFACTS_IMAGES / "Robotic.png"
DOCTOR_PNG = ARTIFACTS_IMAGES / "Doctor.png"

# ──────────────────────────────────────────────────────────────────────
# Emoji constants (Unicode escape codes for portability)
# ──────────────────────────────────────────────────────────────────────
E_GEAR        = "\u2699\uFE0F"       #  gear
E_CHECK       = "\u2705"             #  check mark
E_INFO        = "\u2139\uFE0F"       #  information
E_LINK        = "\U0001F517"         #  link
E_FOLDER_OPEN = "\U0001F4C2"         #  open folder
E_BOOK        = "\U0001F4D6"         #  open book
E_TROPHY      = "\U0001F3C6"         #  trophy
E_STETHOSCOPE = "\U0001FA7A"         #  stethoscope
E_CHART       = "\U0001F4CA"         #  bar chart
E_FOLDER      = "\U0001F4C1"         #  folder
E_PEOPLE      = "\U0001F465"         #  people
E_PERSON      = "\U0001F464"         #  person
E_PLATE       = "\U0001F37D\uFE0F"   #  fork and plate
E_RUNNER      = "\U0001F3C3"         #  runner
E_TARGET      = "\U0001F3AF"         #  target
E_INBOX       = "\U0001F4E5"         #  inbox tray
E_DOWN_ARROW  = "\u2B07\uFE0F"       #  down arrow
E_OUTBOX      = "\U0001F4E4"         #  outbox tray
E_CLIPBOARD   = "\U0001F4CB"         #  clipboard
E_CROSS       = "\u274C"             #  cross mark
E_LABEL       = "\U0001F3F7\uFE0F"   #  label
E_COPYRIGHT  = "\u00A9\uFE0F"       #  copyright

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Obesity Risk Predictor",
    page_icon=str(ROBOTIC_PNG),
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

# Ordinal survey features — originally discrete survey responses,
# stored as floats in the synthetic dataset. Displayed as dropdowns.
ORDINAL_FEATURES = {
    "FCVC": {
        "label": "How often do you eat vegetables in your meals?",
        "options": {"Never": 1.0, "Sometimes": 2.0, "Always": 3.0},
        "default": "Sometimes",
    },
    "NCP": {
        "label": "How many main meals do you have daily?",
        "options": {"Between 1 and 2": 1.0, "3": 3.0, "More than 3": 4.0},
        "default": "3",
    },
    "CH2O": {
        "label": "How much water do you drink daily?",
        "options": {"Less than a litre": 1.0, "Between 1 and 2 litres": 2.0, "More than 2 litres": 3.0},
        "default": "Between 1 and 2 litres",
    },
    "FAF": {
        "label": "How often do you have physical activity? (weekly)",
        "options": {"I do not have": 0.0, "1 or 2 days": 1.0, "2 or 4 days": 2.0, "4 or 5 days": 3.0},
        "default": "1 or 2 days",
    },
    "TUE": {
        "label": "How much time do you use technology devices daily?",
        "options": {"0-2 hours": 0.0, "3-5 hours": 1.0, "More than 5 hours": 2.0},
        "default": "3-5 hours",
    },
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
    return pd.DataFrame(records, columns=FEATURE_NAMES)


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
    st.image(str(DOCTOR_PNG), width=270)
    st.title(f"{E_GEAR} Configuration")

    # ---- Model selector ----
    st.subheader("Select Model")
    model_names = list(MODEL_REGISTRY.keys())
    default_idx = model_names.index(BEST_MODEL_NAME) if BEST_MODEL_NAME in model_names else 0
    selected_model_name = st.selectbox(
        "Choose classifier",
        model_names,
        index=default_idx,
        help="Choose a trained classifier. The best model is pre-selected based on validation F1 & accuracy.",
    )
    if selected_model_name == BEST_MODEL_NAME:
        st.success(f"{E_CHECK} **Best Model** — {BEST_MODEL_NAME}")
    model = load_model(MODEL_REGISTRY[selected_model_name])

    st.divider()

    # ---- About section ----
    st.markdown(f"##### {E_INFO} About")
    st.markdown(
        "This dashboard predicts obesity risk across **7 categories** using "
        "6 ML models trained on demographic, dietary, and lifestyle survey data "
        "(20,758 samples)."
    )

    st.divider()

    # ---- Useful links ----
    st.markdown(f"##### {E_LINK} Links")
    st.markdown(
        f"- [{E_FOLDER_OPEN} GitHub Repository](https://github.com/karthick-ai-ml/ml-multiclass-classification)\n"
        f"- [{E_BOOK} README & Documentation](https://github.com/karthick-ai-ml/ml-multiclass-classification/blob/main/README.md)\n"
        f"- [{E_TROPHY} Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s4e2)"
    )

# ──────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────
tab_predict, tab_models, tab_dataset = st.tabs([
    f"{E_STETHOSCOPE} Diagnosis",
    f"{E_CHART} Performance",
    f"{E_FOLDER} Dataset",
])

# ======================================================================
# TAB 1 — Diagnosis
# ======================================================================
with tab_predict:
    st.header("Obesity Risk Diagnosis")
    st.markdown(
        "Predict an individual's obesity risk level using machine-learning models trained on "
        "eating habits, physical activity, and lifestyle survey data."
    )

    sub_population, sub_individual = st.tabs([
        f"{E_PEOPLE} Population Screening",
        f"{E_PERSON} Individual Assessment",
    ])

    # ==================================================================
    # SUB-TAB A — Individual Patient Assessment
    # ==================================================================
    with sub_individual:
        st.subheader("Individual Obesity Risk Assessment")
        st.markdown(
            "Complete the health questionnaire below covering **personal metrics**, "
            "**eating habits**, and **lifestyle factors** - then click **Diagnose** "
            "to receive an AI-predicted obesity risk category with confidence scores."
        )

        col_left, col_mid, col_right = st.columns(3)

        input_values: dict = {}

        # ---- Column 1: Personal Information ----
        with col_left:
            st.markdown(f"##### {E_PERSON} Personal Information")
            input_values["Gender"] = st.selectbox(
                "Gender",
                ["Female", "Male"],
                index=0,
                key="manual_Gender",
            )
            input_values["Age"] = st.slider(
                "Age (years)",
                min_value=14.0,
                max_value=61.0,
                value=25.0,
                step=1.0,
                key="manual_Age",
            )
            height_cm = st.slider(
                "Height (cm)",
                min_value=145.0,
                max_value=198.0,
                value=170.0,
                step=1.0,
                key="manual_Height",
            )
            input_values["Height"] = height_cm / 100.0  # convert to metres for model
            input_values["Weight"] = st.slider(
                "Weight (kg)",
                min_value=39.0,
                max_value=165.0,
                value=80.0,
                step=1.0,
                key="manual_Weight",
            )
            input_values["family_history_with_overweight"] = st.selectbox(
                "Family history of overweight?",
                ["no", "yes"],
                index=1,
                key="manual_family_history_with_overweight",
            )

        # ---- Column 2: Eating Habits ----
        with col_mid:
            st.markdown(f"##### {E_PLATE} Eating Habits")
            input_values["FAVC"] = st.selectbox(
                "Do you eat high-caloric food frequently?",
                ["no", "yes"],
                index=1,
                key="manual_FAVC",
            )
            for feat in ["FCVC", "NCP", "CH2O"]:
                info = ORDINAL_FEATURES[feat]
                options = list(info["options"].keys())
                default_idx = options.index(info["default"]) if info["default"] in options else 0
                choice = st.selectbox(
                    info["label"],
                    options,
                    index=default_idx,
                    key=f"manual_{feat}",
                )
                input_values[feat] = info["options"][choice]
            input_values["CAEC"] = st.selectbox(
                "Do you eat food between meals?",
                ["Always", "Frequently", "Sometimes", "no"],
                index=2,
                key="manual_CAEC",
            )
            input_values["CALC"] = st.selectbox(
                "How often do you drink alcohol?",
                ["Frequently", "Sometimes", "no"],
                index=2,
                key="manual_CALC",
            )

        # ---- Column 3: Lifestyle ----
        with col_right:
            st.markdown(f"##### {E_RUNNER} Lifestyle & Habits")
            for feat in ["FAF", "TUE"]:
                info = ORDINAL_FEATURES[feat]
                options = list(info["options"].keys())
                default_idx = options.index(info["default"]) if info["default"] in options else 0
                choice = st.selectbox(
                    info["label"],
                    options,
                    index=default_idx,
                    key=f"manual_{feat}",
                )
                input_values[feat] = info["options"][choice]
            input_values["SMOKE"] = st.selectbox(
                "Do you smoke?",
                ["no", "yes"],
                index=0,
                key="manual_SMOKE",
            )
            input_values["SCC"] = st.selectbox(
                "Do you monitor calorie consumption?",
                ["no", "yes"],
                index=0,
                key="manual_SCC",
            )
            input_values["MTRANS"] = st.selectbox(
                "What transportation do you usually use?",
                ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"],
                index=3,
                key="manual_MTRANS",
            )

        if st.button(f"{E_STETHOSCOPE} Diagnose", type="primary", use_container_width=True):
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
                        <h2 style="color:{colour}; margin:0;">{E_TARGET} {pred_class.replace('_', ' ')}</h2>
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
                    st.plotly_chart(fig, width="stretch")

    # ==================================================================
    # SUB-TAB B — Population Screening (Bulk CSV)
    # ==================================================================
    with sub_population:
        st.subheader("Bulk Population Screening")
        st.markdown(
            "Upload a CSV containing survey responses for **multiple individuals** to screen "
            "an entire cohort for obesity risk levels. Ideal for public health studies or "
            "clinical research analysis."
        )

        # ---- Download sample test CSV ----
        st.markdown(f"##### {E_INBOX} Sample Test Data")
        st.markdown(
            "Download the sample CSV to see the expected format - then upload your own data below."
        )
        if TEST_CSV.exists():
            test_bytes = TEST_CSV.read_bytes()
            st.download_button(
                label=f"{E_DOWN_ARROW}  Download Sample CSV (kaggle_obesity_prediction_test.csv)",
                data=test_bytes,
                file_name="kaggle_obesity_prediction_test.csv",
                mime="text/csv",
            )
        else:
            st.warning("Sample test CSV not found at expected path.")

        st.divider()

        # ---- Upload CSV ----
        st.markdown(f"##### {E_OUTBOX} Upload Survey Data")
        uploaded_file = st.file_uploader(
            "Upload a CSV with raw survey features",
            type=["csv"],
            help="Required columns: Gender, Age, Height, Weight, family_history_with_overweight, "
                 "FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS. "
                 "Optional: 'id' (patient identifier) and 'NObeyesdad' (ground-truth label for accuracy evaluation).",
        )

        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.info(f"{E_CLIPBOARD} **{len(df_upload):,}** records loaded for screening.")
            with st.expander("Preview uploaded data (first 10 rows)", expanded=False):
                st.dataframe(df_upload.head(10), width="stretch")

            required_cols = set(RAW_NUMERICAL) | set(RAW_BINARY.keys()) | set(RAW_CATEGORICAL.keys())
            missing = required_cols - set(df_upload.columns)
            if missing:
                st.error(f"{E_CROSS} Missing required columns: **{', '.join(sorted(missing))}**")
            else:
                with st.spinner("Screening population for obesity risk…"):
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

                # ---- Summary metrics ----
                st.markdown(f"##### {E_CHART} Screening Summary")
                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Total Screened", f"{len(df_upload):,}")
                if has_actual:
                    sm2.metric("Accuracy vs Ground Truth", f"{accuracy:.2%}")
                else:
                    sm2.metric("Avg Confidence", f"{df_upload['Confidence (%)'].mean():.1f}%" if "Confidence (%)" in df_upload.columns else "N/A")
                # Count high-risk individuals (Obesity Type I/II/III)
                high_risk = df_upload["Predicted_Class"].isin(["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]).sum()
                sm3.metric("High Risk (Obesity I–III)", f"{high_risk:,}")
                sm4.metric("High Risk %", f"{high_risk / len(df_upload) * 100:.1f}%")

                # ---- Distribution of predicted classes ----
                st.markdown(f"##### {E_LABEL} Predicted Risk Distribution")
                pred_dist = df_upload["Predicted_Class"].value_counts().reset_index()
                pred_dist.columns = ["Risk Category", "Count"]
                fig_dist = px.bar(
                    pred_dist.sort_values("Risk Category"),
                    x="Risk Category",
                    y="Count",
                    color="Risk Category",
                    color_discrete_map=CLASS_COLORS,
                    text="Count",
                )
                fig_dist.update_traces(textposition="outside")
                fig_dist.update_layout(showlegend=False, height=440, margin=dict(t=40, b=0), xaxis_tickangle=-30)
                st.plotly_chart(fig_dist, width="stretch")

                st.divider()

                # ---- Full results table ----
                st.markdown(f"##### {E_CLIPBOARD} Detailed Screening Results")
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
                    width="stretch",
                    height=400,
                )

                # Download predictions
                csv_out = df_upload.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"{E_DOWN_ARROW}  Download Screening Results CSV",
                    data=csv_out,
                    file_name="obesity_screening_results.csv",
                    mime="text/csv",
                )


# ======================================================================
# TAB 2 — Performance
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
    st.dataframe(val_df_display, width="stretch", hide_index=True)

    # ---- Training metrics comparison ----
    st.subheader("Training Metrics Comparison")
    train_df = pd.DataFrame(eval_results["training_metrics"])
    train_df_display = train_df.copy()
    for c in train_df_display.columns:
        if c != "Model":
            train_df_display[c] = train_df_display[c].apply(lambda x: f"{x:.4f}")
    st.dataframe(train_df_display, width="stretch", hide_index=True)

    # ---- Model observations ----
    st.subheader("Model Observations")
    obs_df = pd.DataFrame(eval_results["model_observations"])
    st.dataframe(obs_df, width="stretch", hide_index=True)

    st.divider()

    # ---- Best model summary ----
    st.subheader(f"{E_TROPHY} Best Model: {BEST_MODEL_NAME}")
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
    st.plotly_chart(fig_cmp, width="stretch")

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
    st.plotly_chart(fig_radar, width="stretch")

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
# TAB 3 — Dataset
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
    fig_target.update_layout(showlegend=False, height=480, margin=dict(t=40, b=0))
    st.plotly_chart(fig_target, width="stretch")

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
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

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
        st.plotly_chart(fig_cat, width="stretch")

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
        st.dataframe(load_csv(str(sample_head)), width="stretch", hide_index=True)

    st.divider()

    # ---- Preprocessing information ----
    st.subheader("Preprocessing Summary")
    pp = preprocess_analysis
    pp1, pp2, pp3 = st.columns(3)
    pp1.metric("Train Shape", f"{pp['train_shape'][0]} x {pp['train_shape'][1]}")
    pp2.metric("Validation Shape", f"{pp['validation_shape'][0]} x {pp['validation_shape'][1]}")
    pp3.metric("Test Shape", f"{pp['test_shape'][0]} x {pp['test_shape'][1]}")

    st.markdown("**Encoding Summary**")
    enc_rows = []
    for feat, info in pp["preprocessing_stats"]["encoding"].items():
        enc_rows.append({
            "Feature": feat,
            "Encoding": info["encoding_type"],
            "New Columns": ", ".join(info["new_columns"]),
            "Num Categories": info["num_categories"],
        })
    st.dataframe(pd.DataFrame(enc_rows), width="stretch", hide_index=True)

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
        Karthick AI ML<br>
        {E_COPYRIGHT} 2026
    </div>
    """,
    unsafe_allow_html=True,
)
