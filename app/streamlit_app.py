from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_logistic_regression_model.joblib"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.json"
TEST_DATA_PATH = BASE_DIR / "data" / "raw" / "mlran" / "MLRan_X_test_RFE.csv"


st.set_page_config(
    page_title="AI Ransomware Detection",
    page_icon="",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    :root {
        --bg-main: #030711;
        --bg-card: rgba(10, 20, 35, 0.78);
        --cyan: #00E5FF;
        --blue: #2563EB;
        --violet: #8B5CF6;
        --green: #22C55E;
        --red: #FF3B6B;
        --text-main: #F8FAFC;
        --text-muted: #94A3B8;
        --border-glow: rgba(0, 229, 255, 0.35);
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 8%, rgba(0, 229, 255, 0.16), transparent 28%),
            radial-gradient(circle at 88% 10%, rgba(139, 92, 246, 0.18), transparent 28%),
            radial-gradient(circle at 55% 95%, rgba(37, 99, 235, 0.16), transparent 32%),
            linear-gradient(135deg, #030711 0%, #050B18 45%, #07111F 100%);
        color: var(--text-main);
    }

    header[data-testid="stHeader"] {
        background: rgba(3, 7, 17, 0.15);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 30% 10%, rgba(0, 229, 255, 0.18), transparent 28%),
            linear-gradient(180deg, rgba(5, 11, 24, 0.98), rgba(3, 7, 17, 0.98));
        border-right: 1px solid rgba(0, 229, 255, 0.22);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-main);
    }

    .side-label {
        display: inline-block;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--cyan);
        border: 1px solid rgba(0, 229, 255, 0.35);
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        margin-bottom: 0.85rem;
        background: rgba(0, 229, 255, 0.08);
    }

    .side-title {
        font-size: 1.10rem;
        font-weight: 850;
        line-height: 1.25;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #F8FAFC, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .side-card {
        border: 1px solid rgba(0, 229, 255, 0.22);
        background: rgba(10, 20, 35, 0.62);
        border-radius: 18px;
        padding: 1rem;
        margin: 0.85rem 0;
        box-shadow: 0 0 22px rgba(0, 229, 255, 0.08);
    }

    .side-key {
        font-size: 0.70rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.15rem;
    }

    .side-value {
        font-size: 0.88rem;
        color: var(--text-main);
        margin-bottom: 0.65rem;
        font-weight: 650;
    }

    .hero-card {
        border: 1px solid rgba(0, 229, 255, 0.28);
        background:
            linear-gradient(135deg, rgba(10, 20, 35, 0.90), rgba(15, 28, 48, 0.56)),
            radial-gradient(circle at 85% 20%, rgba(139, 92, 246, 0.20), transparent 30%),
            radial-gradient(circle at 18% 80%, rgba(0, 229, 255, 0.14), transparent 34%);
        border-radius: 28px;
        padding: 2rem 2.1rem;
        box-shadow:
            0 0 0 1px rgba(255, 255, 255, 0.03) inset,
            0 24px 80px rgba(0, 0, 0, 0.35),
            0 0 42px rgba(0, 229, 255, 0.08);
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        display: inline-block;
        color: var(--cyan);
        border: 1px solid rgba(0, 229, 255, 0.35);
        background: rgba(0, 229, 255, 0.08);
        border-radius: 999px;
        padding: 0.32rem 0.78rem;
        font-size: 0.72rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 4rem);
        line-height: 1.02;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin: 0;
        background: linear-gradient(90deg, #F8FAFC 0%, #DDF8FF 45%, #00E5FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: var(--text-muted);
        font-size: 1.04rem;
        max-width: 850px;
        margin-top: 1rem;
        line-height: 1.65;
    }

    .section-kicker {
        color: var(--cyan);
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 850;
        letter-spacing: 0.16em;
        margin-top: 1.4rem;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 850;
        color: var(--text-main);
        margin: 0.25rem 0 0.85rem 0;
    }

    .glass-card {
        border: 1px solid rgba(0, 229, 255, 0.22);
        background: linear-gradient(145deg, rgba(10, 20, 35, 0.84), rgba(15, 28, 48, 0.55));
        border-radius: 22px;
        padding: 1.05rem;
        min-height: 122px;
        box-shadow:
            0 0 0 1px rgba(255, 255, 255, 0.03) inset,
            0 18px 50px rgba(0, 0, 0, 0.22),
            0 0 26px rgba(0, 229, 255, 0.06);
        margin-bottom: 1rem;
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        margin-bottom: 0.42rem;
        font-weight: 800;
    }

    .metric-value {
        color: var(--text-main);
        font-size: 1.15rem;
        font-weight: 850;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .metric-value-cyan {
        color: var(--cyan);
        text-shadow: 0 0 18px rgba(0, 229, 255, 0.35);
    }

    .note-card {
        border-left: 4px solid var(--cyan);
        border-top: 1px solid rgba(0, 229, 255, 0.22);
        border-right: 1px solid rgba(0, 229, 255, 0.18);
        border-bottom: 1px solid rgba(0, 229, 255, 0.18);
        background: rgba(0, 229, 255, 0.07);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        color: #DDF8FF;
        margin: 1rem 0 1.4rem 0;
    }

    .result-good {
        border: 1px solid rgba(34, 197, 94, 0.38);
        background: rgba(34, 197, 94, 0.08);
        color: #BBF7D0;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        font-weight: 800;
        margin: 1rem 0;
    }

    .result-bad {
        border: 1px solid rgba(255, 59, 107, 0.40);
        background: rgba(255, 59, 107, 0.08);
        color: #FFD1DC;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        font-weight: 800;
        margin: 1rem 0;
    }

    .small-muted {
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .footer {
        border-top: 1px solid rgba(0, 229, 255, 0.18);
        color: var(--text-muted);
        margin-top: 2.4rem;
        padding-top: 1rem;
        font-size: 0.85rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(10, 20, 35, 0.60);
        border: 1px solid rgba(0, 229, 255, 0.20);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 0 22px rgba(0, 229, 255, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--text-main);
        font-weight: 900;
    }

    .stDataFrame {
        border: 1px solid rgba(0, 229, 255, 0.18);
        border-radius: 16px;
        overflow: hidden;
    }
</style>
"""


def render_metric_card(label: str, value: str, accent: bool = False):
    accent_class = " metric-value-cyan" if accent else ""
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value{accent_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(kicker: str, title: str):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_columns():
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_DATA_PATH)


def label_to_text(label: int) -> str:
    if label == 0:
        return "Goodware / Benign"
    if label == 1:
        return "Ransomware"
    return "Unknown"


def extract_final_estimator(model_object):
    if hasattr(model_object, "steps"):
        return model_object.steps[-1][1]
    return model_object


def get_feature_importance(model_object, feature_columns):
    estimator = extract_final_estimator(model_object)

    if not hasattr(estimator, "coef_"):
        return None

    coefficients = estimator.coef_[0]

    importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Coefficient": coefficients,
            "Absolute Weight": abs(coefficients),
        }
    )

    return importance_df.sort_values("Absolute Weight", ascending=False).head(10)


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="side-label">Research Prototype</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-title">AI-Driven Behavioral Ransomware Detection System</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-card">
            <div class="side-key">Degree</div>
            <div class="side-value">MSc Cyber Security</div>
            <div class="side-key">Institution</div>
            <div class="side-value">Arden University</div>
            <div class="side-key">Researcher</div>
            <div class="side-value">Keshav Vasalli</div>
            <div class="side-key">Year</div>
            <div class="side-value">2026</div>
            <div class="side-key">Prototype Type</div>
            <div class="side-value">Research Prototype</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Dashboard Sections")
    st.markdown(
        """
        - Overview
        - Model Performance
        - Model Prediction Demo
        - Feature Analysis
        - Selected Sample Features
        - How It Works
        """
    )


st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Research Prototype</div>
        <h1 class="hero-title">AI-Driven Behavioral Ransomware Detection System</h1>
        <div class="hero-subtitle">
            Research prototype for AI-driven behavioural ransomware detection using
            processed behavioural feature data from the MLRan dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="note-card">
        <strong>Safety Note:</strong> This system uses pre-extracted behavioural features only.
        No live ransomware files are uploaded, executed, or analysed within this environment.
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    model = load_model()
    feature_columns = load_feature_columns()
    test_df = load_test_data()

    render_section("Overview", "System Overview")

    overview_cols = st.columns(6)
    overview_items = [
        ("Dataset", "MLRan (Processed)"),
        ("Best Model", "Logistic Regression"),
        ("Total Features", str(len(feature_columns))),
        ("Test Samples", str(len(test_df))),
        ("Task", "Binary Classification"),
        ("Classes", "Goodware/Benign and Ransomware"),
    ]

    for column, (label, value) in zip(overview_cols, overview_items):
        with column:
            render_metric_card(label, value, accent=label in ["Best Model", "Total Features"])

    render_section("Evaluation", "Model Performance")

    performance_cols = st.columns(4)
    performance_items = [
        ("Accuracy", "97.85%"),
        ("Precision", "97.03%"),
        ("Recall", "98.49%"),
        ("F1-score", "97.76%"),
    ]

    for column, (label, value) in zip(performance_cols, performance_items):
        with column:
            render_metric_card(label, value, accent=True)

    st.markdown(
        '<div class="small-muted">Metrics are based on the saved Logistic Regression model evaluation from the MLRan test set.</div>',
        unsafe_allow_html=True,
    )

    render_section("Evaluation Note", "Confusion Matrix")

    st.markdown(
        """
        <div class="note-card">
            Confusion matrix visualisation will be added only after verifying the exact
            test-set values from the model evaluation output. This avoids displaying estimated
            or synthetic confusion matrix values.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section("Prediction", "Model Prediction Demo")

    selected_sample_id = st.selectbox(
        "Choose a sample_id from the MLRan test dataset:",
        test_df["sample_id"].tolist(),
    )

    selected_row = test_df[test_df["sample_id"] == selected_sample_id].iloc[0]
    actual_label = int(selected_row["sample_type"])
    X_sample = selected_row[feature_columns].to_frame().T

    prediction = int(model.predict(X_sample)[0])
    probabilities = model.predict_proba(X_sample)[0]

    benign_probability = probabilities[0]
    ransomware_probability = probabilities[1]
    confidence = max(benign_probability, ransomware_probability)

    prediction_cols = st.columns(3)

    with prediction_cols[0]:
        st.metric("Actual Label", label_to_text(actual_label))

    with prediction_cols[1]:
        st.metric("Predicted Label", label_to_text(prediction))

    with prediction_cols[2]:
        st.metric("Prediction Confidence", f"{confidence:.2%}")

    if prediction == 1:
        st.markdown(
            '<div class="result-bad">Classification result: Ransomware behaviour detected for the selected processed sample.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-good">Classification result: Goodware / benign behaviour detected for the selected processed sample.</div>',
            unsafe_allow_html=True,
        )

    probability_df = pd.DataFrame(
        {
            "Class": ["Goodware / Benign", "Ransomware"],
            "Probability": [benign_probability, ransomware_probability],
        }
    )

    st.markdown("#### Class Probability Scores")

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.metric("Goodware / Benign Probability", f"{benign_probability:.2%}")
        st.progress(float(benign_probability))

    with prob_col2:
        st.metric("Ransomware Probability", f"{ransomware_probability:.2%}")
        st.progress(float(ransomware_probability))

    render_section("Interpretability", "Feature Analysis / Explainability")

    importance_df = get_feature_importance(model, feature_columns)

    if importance_df is None:
        st.markdown(
            """
            <div class="note-card">
                Feature importance is not displayed because the saved model object does not expose
                directly accessible coefficients in the current pipeline structure.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="small-muted">Top model coefficient features from the selected feature vector. Feature weights are shown only where directly accessible from the saved model pipeline.</div>',
            unsafe_allow_html=True,
        )
        styled_importance_df = importance_df[["Feature", "Coefficient", "Absolute Weight"]].copy()

        st.dataframe(
            styled_importance_df,
            width="stretch",
            hide_index=True,
        )

    render_section("Sample Data", "Selected Sample — Behavioural Features")

    with st.expander("Show selected sample metadata", expanded=True):
        metadata_df = selected_row[
            ["sample_id", "sample_type", "family_label", "type_label"]
        ].to_frame("value")
        st.dataframe(metadata_df, width="stretch")

    with st.expander("Show selected behavioural feature values"):
        st.dataframe(X_sample, width="stretch")

    render_section("Workflow", "How It Works")

    workflow_cols = st.columns(5)
    workflow_items = [
        ("01", "Dataset Preparation", "Processed MLRan behavioural feature data is used."),
        ("02", "Pre-processing", "Feature-selected behavioural vectors are loaded for model testing."),
        ("03", "Model Training", "Machine learning models were trained and compared during the project."),
        ("04", "Prediction", "The saved Logistic Regression model classifies selected test samples."),
        ("05", "Analysis / Insights", "The dashboard presents prediction confidence, probabilities, and available feature values."),
    ]

    for column, (number, title, description) in zip(workflow_cols, workflow_items):
        with column:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-label">{number}</div>
                    <div class="metric-value">{title}</div>
                    <div class="small-muted">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_section("Project", "Project Information")

    project_cols = st.columns(5)
    project_items = [
        ("Prototype Type", "Research Prototype"),
        ("Degree", "MSc Cyber Security"),
        ("Institution", "Arden University"),
        ("Researcher", "Keshav Vasalli"),
        ("Year", "2026"),
    ]

    for column, (label, value) in zip(project_cols, project_items):
        with column:
            render_metric_card(label, value)

    st.markdown(
        """
        <div class="footer">
            AI-Driven Behavioral Ransomware Detection System — MSc Cyber Security thesis research prototype.
        </div>
        """,
        unsafe_allow_html=True,
    )

except FileNotFoundError as error:
    st.error("Required project file was not found.")
    st.write(error)
    st.warning(
        "Make sure the saved model, feature columns file, and local MLRan test dataset "
        "exist in the expected project folders."
    )

except Exception as error:
    st.error("An unexpected error occurred while running the app.")
    st.write(error)
