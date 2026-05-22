from pathlib import Path
import json
from html import escape
from textwrap import dedent

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_logistic_regression_model.joblib"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.json"
TEST_DATA_PATH = BASE_DIR / "data" / "raw" / "mlran" / "MLRan_X_test_RFE.csv"


st.set_page_config(
    page_title="AI Ransomware Detection",
    page_icon="🛡️",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    :root {
        --bg-main: #020611;
        --bg-panel: rgba(8, 18, 34, 0.88);
        --bg-panel-soft: rgba(11, 27, 49, 0.72);
        --cyan: #00E5FF;
        --blue: #2F80ED;
        --violet: #8B5CF6;
        --green: #22C55E;
        --red: #FF3B6B;
        --text-main: #F8FAFC;
        --text-soft: #D7E7FF;
        --text-muted: #94A3B8;
        --border: rgba(0, 229, 255, 0.22);
        --border-strong: rgba(0, 229, 255, 0.42);
        --shadow-glow: 0 0 38px rgba(0, 229, 255, 0.10);
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background:
            radial-gradient(circle at 18% 9%, rgba(0, 229, 255, 0.13), transparent 26%),
            radial-gradient(circle at 78% 6%, rgba(139, 92, 246, 0.16), transparent 28%),
            radial-gradient(circle at 60% 92%, rgba(47, 128, 237, 0.10), transparent 34%),
            linear-gradient(135deg, #020611 0%, #040B17 48%, #07101E 100%);
        color: var(--text-main);
    }

    header[data-testid="stHeader"],
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    .block-container {
        max-width: none !important;
        padding-top: 0.75rem !important;
        padding-right: 0.85rem !important;
        padding-bottom: 1rem !important;
        padding-left: 16.25rem !important;
    }

    .fixed-sidebar {
        position: fixed;
        left: 0.75rem;
        top: 0.75rem;
        bottom: 0.75rem;
        width: 14rem;
        z-index: 999;
        border: 1px solid rgba(0, 229, 255, 0.20);
        border-radius: 18px;
        background:
            radial-gradient(circle at 28% 8%, rgba(0, 229, 255, 0.16), transparent 25%),
            linear-gradient(180deg, rgba(7, 16, 31, 0.98), rgba(2, 6, 17, 0.98));
        box-shadow: 0 22px 70px rgba(0, 0, 0, 0.40), var(--shadow-glow);
        padding: 1rem 0.95rem;
        overflow-y: auto;
        overflow-x: hidden;
    }

    .fixed-sidebar::-webkit-scrollbar { width: 0.25rem; }
    .fixed-sidebar::-webkit-scrollbar-thumb { background: rgba(0, 229, 255, 0.22); border-radius: 999px; }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.64rem;
        margin-bottom: 1rem;
    }

    .brand-shield {
        width: 2.55rem;
        height: 2.55rem;
        display: grid;
        place-items: center;
        border-radius: 16px;
        background:
            radial-gradient(circle at 35% 25%, rgba(0, 229, 255, 0.32), transparent 38%),
            linear-gradient(145deg, rgba(17, 34, 64, 0.95), rgba(6, 14, 31, 0.95));
        border: 1px solid rgba(0, 229, 255, 0.34);
        box-shadow: 0 0 28px rgba(0, 229, 255, 0.20);
        color: #E9FBFF;
        font-weight: 900;
        letter-spacing: 0.03em;
    }

    .brand-title {
        font-size: 0.92rem;
        line-height: 1.18;
        font-weight: 900;
        letter-spacing: 0.02em;
        color: var(--text-main);
        text-transform: uppercase;
    }

    .brand-subtitle {
        color: var(--text-muted);
        font-size: 0.68rem;
        line-height: 1.2;
        margin-top: 0.15rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.62rem;
        margin: 0.25rem 0;
        padding: 0.58rem 0.64rem;
        color: #EAF7FF !important;
        font-size: 0.78rem;
        font-weight: 750;
        border-radius: 13px;
        border: 1px solid transparent;
        text-decoration: none !important;
        cursor: pointer;
    }

    .fixed-sidebar a,
    .fixed-sidebar a:visited,
    .fixed-sidebar a:hover,
    .fixed-sidebar a:active {
        color: #EAF7FF !important;
        text-decoration: none !important;
    }

    .nav-item:hover {
        background: rgba(0, 229, 255, 0.10);
        border-color: rgba(0, 229, 255, 0.22);
    }

    .nav-item.active {
        color: #FFFFFF !important;
        background: linear-gradient(90deg, rgba(47, 128, 237, 0.46), rgba(0, 229, 255, 0.16));
        border-color: rgba(0, 229, 255, 0.26);
        box-shadow: inset -3px 0 0 rgba(0, 229, 255, 0.86);
    }

    .nav-icon {
        width: 1.12rem;
        text-align: center;
        color: var(--cyan);
        filter: drop-shadow(0 0 7px rgba(0, 229, 255, 0.45));
    }

    .sidebar-card {
        margin-top: 0.8rem;
        border: 1px solid rgba(0, 229, 255, 0.20);
        border-radius: 16px;
        background: rgba(8, 18, 34, 0.72);
        padding: 0.75rem;
        color: var(--text-soft);
        font-size: 0.70rem;
        line-height: 1.42;
    }

    .sidebar-stack {
        margin-top: 0.75rem;
        border-top: 1px solid rgba(0, 229, 255, 0.14);
        padding-top: 0.65rem;
        color: var(--text-muted);
        font-size: 0.68rem;
        line-height: 1.35;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.85fr) minmax(260px, 0.82fr);
        gap: 0.72rem;
        margin-bottom: 0.72rem;
    }

    .hero-panel, .system-panel {
        min-height: 210px;
        border: 1px solid var(--border);
        border-radius: 18px;
        background:
            linear-gradient(145deg, rgba(8, 18, 34, 0.92), rgba(12, 26, 48, 0.64)),
            radial-gradient(circle at 90% 0%, rgba(0, 229, 255, 0.08), transparent 30%);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28), var(--shadow-glow);
        padding: 1.05rem;
    }

    .eyebrow {
        display: inline-flex;
        width: fit-content;
        color: #B8F7FF;
        background: rgba(68, 56, 202, 0.32);
        border: 1px solid rgba(139, 92, 246, 0.38);
        border-radius: 999px;
        padding: 0.32rem 0.72rem;
        font-size: 0.65rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 850;
        margin-bottom: 0.85rem;
    }

    .hero-title {
        font-size: clamp(1.8rem, 2.65vw, 2.9rem);
        line-height: 1.04;
        margin: 0;
        font-weight: 950;
        letter-spacing: -0.055em;
        color: var(--text-main);
    }

    .hero-title span {
        background: linear-gradient(90deg, #EAF8FF 0%, #A6F2FF 55%, #2F80ED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: var(--text-soft);
        font-size: 0.84rem;
        line-height: 1.45;
        margin-top: 0.72rem;
        max-width: 720px;
    }

    .hero-mini-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
        margin-top: 1.15rem;
    }

    .mini-chip {
        border: 1px solid rgba(0, 229, 255, 0.14);
        background: rgba(6, 14, 31, 0.58);
        border-radius: 14px;
        padding: 0.68rem 0.75rem;
    }

    .chip-label, .sys-label, .metric-name {
        color: var(--text-muted);
        font-size: 0.64rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.28rem;
        font-weight: 850;
    }

    .chip-value, .sys-value {
        color: var(--text-main);
        font-size: 0.86rem;
        font-weight: 850;
    }

    .safety-note {
        margin-top: 1.05rem;
        border: 1px solid rgba(0, 229, 255, 0.22);
        border-radius: 14px;
        background: rgba(0, 229, 255, 0.07);
        color: #DDF8FF;
        padding: 0.75rem;
        font-size: 0.76rem;
        line-height: 1.48;
    }

    .visual-card {
        display: grid;
        place-items: center;
        background:
            radial-gradient(circle, rgba(0, 229, 255, 0.24), transparent 35%),
            radial-gradient(circle, rgba(139, 92, 246, 0.17), transparent 55%),
            linear-gradient(145deg, rgba(8, 18, 34, 0.90), rgba(12, 26, 48, 0.60));
    }

    .orb {
        width: 9.6rem;
        height: 9.6rem;
        border-radius: 50%;
        border: 1px solid rgba(0, 229, 255, 0.25);
        background:
            radial-gradient(circle, rgba(0, 229, 255, 0.28), transparent 34%),
            conic-gradient(from 20deg, transparent, rgba(0, 229, 255, 0.85), rgba(139, 92, 246, 0.82), transparent);
        display: grid;
        place-items: center;
        box-shadow: 0 0 42px rgba(0, 229, 255, 0.22);
    }

    .shield-core {
        width: 3.7rem;
        height: 3.7rem;
        border-radius: 20px;
        display: grid;
        place-items: center;
        color: #B8F7FF;
        font-size: 1.12rem;
        background: linear-gradient(145deg, rgba(13, 35, 65, 0.95), rgba(5, 12, 28, 0.95));
        border: 1px solid rgba(0, 229, 255, 0.32);
        box-shadow: inset 0 0 25px rgba(47, 128, 237, 0.24);
    }

    .system-panel {
        display: flex;
        flex-direction: column;
        gap: 0.58rem;
    }

    .panel-title {
        color: var(--text-main);
        font-weight: 900;
        margin-bottom: 0.18rem;
    }

    .sys-row {
        display: grid;
        grid-template-columns: 1.9rem 1fr;
        gap: 0.55rem;
        align-items: center;
    }

    .sys-icon {
        width: 1.55rem;
        height: 1.55rem;
        display: grid;
        place-items: center;
        border-radius: 8px;
        color: var(--cyan);
        background: rgba(47, 128, 237, 0.12);
        border: 1px solid rgba(0, 229, 255, 0.14);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        background:
            linear-gradient(145deg, rgba(8, 18, 34, 0.90), rgba(12, 26, 48, 0.64)),
            radial-gradient(circle at 90% 0%, rgba(0, 229, 255, 0.08), transparent 30%) !important;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28), var(--shadow-glow) !important;
        padding: 0.62rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: transparent !important;
    }

    .section-title {
        font-size: 1.12rem !important;
        line-height: 1.14;
        margin: 0 0 0.55rem 0;
        color: var(--text-main);
        font-weight: 950;
        letter-spacing: -0.04em;
    }

    .section-sub {
        color: var(--text-muted);
        font-size: 0.80rem;
        line-height: 1.45;
        margin-bottom: 0.8rem;
    }

    .confidence-ring {
        --pct: 80%;
        width: 7rem;
        height: 7rem;
        margin: 0.6rem auto;
        border-radius: 50%;
        background: conic-gradient(var(--green) var(--pct), rgba(148, 163, 184, 0.12) 0);
        display: grid;
        place-items: center;
        box-shadow: 0 0 34px rgba(34, 197, 94, 0.20);
    }

    .ring-inner {
        width: 5.1rem;
        height: 5.1rem;
        border-radius: 50%;
        background: #020611;
        display: grid;
        place-items: center;
        text-align: center;
    }

    .ring-value {
        font-size: 1.15rem;
        font-weight: 950;
        color: var(--text-main);
    }

    .ring-label {
        color: var(--green);
        font-size: 0.68rem;
        font-weight: 850;
    }

    .result-pill {
        border: 1px solid rgba(34, 197, 94, 0.34);
        background: rgba(34, 197, 94, 0.12);
        color: #BBF7D0;
        border-radius: 12px;
        padding: 0.82rem 0.9rem;
        font-weight: 900;
        margin: 0.55rem 0 0.9rem 0;
    }

    .result-pill.bad {
        border-color: rgba(255, 59, 107, 0.42);
        background: rgba(255, 59, 107, 0.13);
        color: #FFD1DC;
    }

    .prob-row {
        display: grid;
        grid-template-columns: 8.2rem 4.4rem 1fr;
        gap: 0.55rem;
        align-items: center;
        color: var(--text-soft);
        font-size: 0.78rem;
        margin: 0.55rem 0;
    }

    .bar-track, .feature-track {
        height: 0.48rem;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.14);
        overflow: hidden;
    }

    .bar-fill, .feature-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #22C55E, #00E5FF);
    }

    .bar-fill.red {
        background: linear-gradient(90deg, #FF3B6B, #F59E0B);
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
    }

    .metric-card {
        border: 1px solid rgba(0, 229, 255, 0.14);
        background: linear-gradient(145deg, rgba(14, 34, 63, 0.75), rgba(6, 14, 31, 0.70));
        border-radius: 14px;
        padding: 0.68rem;
    }

    .metric-number {
        color: var(--text-main);
        font-size: 1.12rem;
        font-weight: 950;
    }

    .spark {
        height: 1.2rem;
        margin-top: 0.5rem;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(47, 128, 237, 0.15), rgba(0, 229, 255, 0.28));
    }

    .cmatrix {
        width: 100%;
        border-collapse: collapse;
        color: var(--text-soft);
        overflow: hidden;
        border-radius: 12px;
        font-size: 0.78rem;
    }

    .cmatrix th, .cmatrix td {
        border: 1px solid rgba(0, 229, 255, 0.14);
        padding: 0.72rem;
        text-align: center;
    }

    .cmatrix th {
        background: rgba(47, 128, 237, 0.20);
        color: var(--text-main);
        font-weight: 900;
    }

    .feature-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 1.6fr 3.5rem;
        gap: 0.55rem;
        align-items: center;
        margin: 0.45rem 0;
        color: var(--text-soft);
        font-size: 0.76rem;
    }

    .feature-name {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .feature-fill {
        background: linear-gradient(90deg, #635BFF, #8B5CF6, #00E5FF);
    }

    .reason-list, .info-list {
        list-style: none;
        padding: 0;
        margin: 0;
        color: var(--text-soft);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .reason-list li {
        display: grid;
        grid-template-columns: 1.4rem 1fr;
        gap: 0.45rem;
        margin: 0.62rem 0;
    }

    .check {
        color: var(--green);
        font-weight: 950;
    }

    .flow-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.65rem;
    }

    .flow-card {
        border: 1px solid rgba(0, 229, 255, 0.14);
        background: rgba(6, 14, 31, 0.52);
        border-radius: 14px;
        padding: 0.8rem;
        min-height: 6rem;
    }

    .flow-icon {
        width: 2rem;
        height: 2rem;
        display: grid;
        place-items: center;
        border-radius: 12px;
        color: var(--cyan);
        background: rgba(0, 229, 255, 0.09);
        border: 1px solid rgba(0, 229, 255, 0.22);
        margin-bottom: 0.55rem;
        font-size: 1rem;
    }

    .flow-number {
        color: var(--cyan);
        font-weight: 950;
        margin-bottom: 0.35rem;
    }

    .flow-title {
        color: var(--text-main);
        font-weight: 900;
        font-size: 0.82rem;
        margin-bottom: 0.28rem;
    }

    .flow-copy {
        color: var(--text-muted);
        font-size: 0.72rem;
        line-height: 1.45;
    }

    div[data-testid="stMetric"] {
        background: rgba(6, 14, 31, 0.52);
        border: 1px solid rgba(0, 229, 255, 0.14);
        border-radius: 14px;
        padding: 0.8rem;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text-main);
        font-weight: 950;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
    }

    .stDataFrame {
        border: 1px solid rgba(0, 229, 255, 0.16);
        border-radius: 12px;
        overflow: hidden;
    }



    div[data-baseweb="select"] > div {
        background: rgba(5, 13, 28, 0.92) !important;
        border: 1px solid rgba(0, 229, 255, 0.26) !important;
        border-radius: 12px !important;
        color: var(--text-main) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: var(--text-main) !important;
    }

    .stSelectbox label, .stMetric label {
        color: var(--text-muted) !important;
    }

    #dashboard, #live-detection, #model-performance, #confusion-matrix,
    #feature-analysis, #how-it-works, #about-project {
        scroll-margin-top: 1rem;
    }

    @media (max-width: 1150px) {
        .block-container { padding-left: 1rem !important; }
        .fixed-sidebar { display: none; }
        .hero-grid { grid-template-columns: 1fr; }
        .metric-grid, .flow-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
</style>
"""


def render_html(markup: str) -> None:
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def render_sidebar() -> None:
    render_html(
        """
        <div class="fixed-sidebar">
            <div class="brand-row">
                <div class="brand-shield">AI</div>
                <div>
                    <div class="brand-title">AI-Driven<br>Behavioral</div>
                    <div class="brand-subtitle">Ransomware Detection System</div>
                </div>
            </div>
            <a class="nav-item active" href="#dashboard"><span class="nav-icon">▣</span><span>Dashboard</span></a>
            <a class="nav-item" href="#live-detection"><span class="nav-icon">◉</span><span>Live Detection</span></a>
            <a class="nav-item" href="#model-performance"><span class="nav-icon">↗</span><span>Model Performance</span></a>
            <a class="nav-item" href="#confusion-matrix"><span class="nav-icon">▦</span><span>Confusion Matrix</span></a>
            <a class="nav-item" href="#feature-analysis"><span class="nav-icon">✧</span><span>Feature Analysis</span></a>
            <a class="nav-item" href="#how-it-works"><span class="nav-icon">◆</span><span>How It Works</span></a>
            <a class="nav-item" href="#about-project"><span class="nav-icon">ⓘ</span><span>About Project</span></a>
            <div class="sidebar-card"><strong>About This Project</strong><br><br>MSc Cyber Security research prototype at Arden University.</div>
            <div class="sidebar-stack"><strong>Tech Stack</strong><br>Python · Scikit-learn · Streamlit<br>Pandas · Joblib · MLRan<br><br>© 2026 | MSc Cyber Security<br>Arden University</div>
        </div>
        """
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_columns():
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_test_data():
    dataframe = pd.read_csv(TEST_DATA_PATH)
    if "sample_id" in dataframe.columns:
        dataframe["sample_id"] = dataframe["sample_id"].astype(str).str.zfill(5)
    return dataframe


def label_to_text(label: int) -> str:
    if label == 0:
        return "Benign (Goodware)"
    if label == 1:
        return "Ransomware"
    return "Unknown"


def safe_value(value) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


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


def calculate_metrics(model_object, test_df, feature_columns):
    x_test = test_df[feature_columns]
    y_true = test_df["sample_type"].astype(int)
    y_pred = pd.Series(model_object.predict(x_test)).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total = max(tn + fp + fn + tp, 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def get_class_probabilities(model_object, x_sample):
    probabilities = model_object.predict_proba(x_sample)[0]
    estimator = extract_final_estimator(model_object)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    benign_probability = probabilities[classes.index(0)] if 0 in classes else probabilities[0]
    ransomware_probability = probabilities[classes.index(1)] if 1 in classes else probabilities[1]
    return float(benign_probability), float(ransomware_probability)


def render_hero(total_features: int, total_samples: int) -> None:
    render_html(
        f"""
        <div class="hero-grid">
            <div class="hero-panel">
                <div class="eyebrow">Research Prototype</div>
                <h1 class="hero-title">AI-Driven Behavioral<br><span>Ransomware Detection System</span></h1>
                <div class="hero-subtitle">Machine learning based classification of files as benign/goodware or ransomware using processed behavioural feature data from the MLRan dataset.</div>
                <div class="hero-mini-grid">
                    <div class="mini-chip"><div class="chip-label">Dataset</div><div class="chip-value">MLRan (Processed)</div></div>
                    <div class="mini-chip"><div class="chip-label">Best Model</div><div class="chip-value">Logistic Regression</div></div>
                </div>
                <div class="safety-note"><strong>Safety Note:</strong> This dashboard uses pre-extracted behavioural features only. No live ransomware files are uploaded, executed, or analysed.</div>
            </div>
            <div class="system-panel">
                <div class="panel-title">System Overview</div>
                <div class="sys-row"><div class="sys-icon">⌘</div><div><div class="sys-label">Total Features</div><div class="sys-value">{total_features}</div></div></div>
                <div class="sys-row"><div class="sys-icon">▣</div><div><div class="sys-label">Total Test Samples</div><div class="sys-value">{total_samples}</div></div></div>
                <div class="sys-row"><div class="sys-icon">◎</div><div><div class="sys-label">Model Type</div><div class="sys-value">Logistic Regression</div></div></div>
                <div class="sys-row"><div class="sys-icon">⌬</div><div><div class="sys-label">Trained On</div><div class="sys-value">MLRan (Processed)</div></div></div>
                <div class="sys-row"><div class="sys-icon">◇</div><div><div class="sys-label">Task</div><div class="sys-value">Binary Classification</div></div></div>
                <div class="sys-row"><div class="sys-icon">⟡</div><div><div class="sys-label">Classes</div><div class="sys-value">Benign / Ransomware</div></div></div>
            </div>
        </div>
        """
    )


def section_heading(title: str, subtitle: str = "") -> None:
    subtitle_html = f'<div class="section-sub">{escape(subtitle)}</div>' if subtitle else ""
    render_html(
        f"""
        <div>
            <h2 class="section-title">{escape(title)}</h2>
            {subtitle_html}
        </div>
        """
    )


def render_confidence_ring(confidence: float) -> None:
    confidence_pct = max(0, min(confidence * 100, 100))
    confidence_label = "High Confidence" if confidence >= 0.80 else "Moderate Confidence"
    render_html(
        f"""
        <div class="confidence-ring" style="--pct: {confidence_pct:.2f}%;">
            <div class="ring-inner"><div><div class="ring-value">{confidence_pct:.1f}%</div><div class="ring-label">{confidence_label}</div></div></div>
        </div>
        """
    )


def render_probability_rows(benign_probability: float, ransomware_probability: float) -> None:
    benign_width = benign_probability * 100
    ransomware_width = ransomware_probability * 100
    render_html(
        f"""
        <div class="prob-row"><div>🟢 Benign</div><div>{benign_width:.1f}%</div><div class="bar-track"><div class="bar-fill" style="width: {benign_width:.2f}%"></div></div></div>
        <div class="prob-row"><div>🔴 Ransomware</div><div>{ransomware_width:.1f}%</div><div class="bar-track"><div class="bar-fill red" style="width: {ransomware_width:.2f}%"></div></div></div>
        """
    )


def render_metric_grid(metrics) -> None:
    metric_items = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1-score", metrics["f1"]),
    ]
    cards = ""
    for label, value in metric_items:
        cards += f'<div class="metric-card"><div class="metric-name">{escape(label)}</div><div class="metric-number">{percentage(value)}</div><div class="spark"></div></div>'
    render_html(f'<div class="metric-grid">{cards}</div>')


def render_confusion_matrix(metrics) -> None:
    tn = metrics["tn"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    tp = metrics["tp"]
    benign_total = tn + fp
    ransomware_total = fn + tp
    pred_benign_total = tn + fn
    pred_ransomware_total = fp + tp
    total = benign_total + ransomware_total
    render_html(
        f"""
        <table class="cmatrix">
            <tr><th>Actual \\ Predicted</th><th>Benign</th><th>Ransomware</th><th>Total</th></tr>
            <tr><th>Benign</th><td>{tn}</td><td>{fp}</td><td>{benign_total}</td></tr>
            <tr><th>Ransomware</th><td>{fn}</td><td>{tp}</td><td>{ransomware_total}</td></tr>
            <tr><th>Total</th><td>{pred_benign_total}</td><td>{pred_ransomware_total}</td><td>{total}</td></tr>
        </table>
        """
    )


def render_feature_bars(importance_df) -> None:
    if importance_df is None or importance_df.empty:
        st.info("Feature importance is not available from the saved model pipeline.")
        return
    max_weight = max(float(importance_df["Absolute Weight"].max()), 1e-12)
    rows_html = ""
    for _, row in importance_df.iterrows():
        feature = escape(str(row["Feature"]))
        coeff = float(row["Coefficient"])
        weight = float(row["Absolute Weight"])
        width = max((weight / max_weight) * 100, 2)
        rows_html += f'<div class="feature-row"><div class="feature-name" title="{feature}">{feature}</div><div class="feature-track"><div class="feature-fill" style="width: {width:.2f}%"></div></div><div class="feature-value">{coeff:.3f}</div></div>'
    render_html(rows_html)


def build_sample_feature_preview(importance_df, selected_row):
    if importance_df is None or importance_df.empty:
        return pd.DataFrame()
    preview_features = importance_df["Feature"].head(6).tolist()
    return pd.DataFrame(
        {
            "Feature": preview_features,
            "Value": [safe_value(selected_row.get(feature, "-")) for feature in preview_features],
        }
    )


def render_prediction_reasons(prediction: int, confidence: float, importance_df, selected_row) -> None:
    predicted_label = label_to_text(prediction)
    reasons = [
        f"The highest predicted class probability is {predicted_label} at {confidence:.1%}.",
        "The result is produced from processed behavioural feature values, not from live malware execution.",
        "The displayed feature analysis uses coefficients from the saved Logistic Regression model where accessible.",
        "The sample is compared against learned benign and ransomware behavioural patterns from the MLRan dataset.",
    ]
    if importance_df is not None and not importance_df.empty:
        top_feature = str(importance_df.iloc[0]["Feature"])
        top_value = safe_value(selected_row.get(top_feature, "-"))
        reasons.insert(2, f"Top weighted model feature shown: {top_feature} = {top_value} for this selected sample.")
    list_items = "".join(f'<li><span class="check">✓</span><span>{escape(reason)}</span></li>' for reason in reasons)
    render_html(f'<ul class="reason-list">{list_items}</ul>')


def render_workflow() -> None:
    workflow_items = [
        ("1", "▧", "Data Collection", "Behavioural features extracted from files in a controlled dataset."),
        ("2", "⚙", "Preprocessing", "Feature-selected data prepared for model training and testing."),
        ("3", "⌁", "Model Training", "Machine learning models compared, with Logistic Regression selected."),
        ("4", "◎", "Prediction", "Selected test samples classified as benign or ransomware."),
        ("5", "✦", "Analysis", "Confidence, metrics and feature signals presented for interpretation."),
    ]
    cards = ""
    for number, icon, title, description in workflow_items:
        cards += f'<div class="flow-card"><div class="flow-icon">{escape(icon)}</div><div class="flow-number">{escape(number)}</div><div class="flow-title">{escape(title)}</div><div class="flow-copy">{escape(description)}</div></div>'
    render_html(f'<div class="flow-grid">{cards}</div>')


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
render_sidebar()

try:
    model = load_model()
    feature_columns = load_feature_columns()
    test_df = load_test_data()

    metrics = calculate_metrics(model, test_df, feature_columns)
    importance_df = get_feature_importance(model, feature_columns)

    render_html('<div id="dashboard"></div>')
    render_hero(total_features=len(feature_columns), total_samples=len(test_df))

    render_html('<div id="live-detection"></div>')
    live_left, live_mid, live_right = st.columns([1.35, 0.92, 1.65], gap="medium")

    with live_left:
        with st.container(border=True):
            section_heading("Live Detection Demo", "Select a processed test sample and view the model prediction.")
            selected_sample_id = st.selectbox(
                "Select Sample ID",
                test_df["sample_id"].tolist(),
                label_visibility="visible",
            )
            selected_row = test_df[test_df["sample_id"] == selected_sample_id].iloc[0]
            actual_label = int(selected_row["sample_type"])
            x_sample = selected_row[feature_columns].to_frame().T
            prediction = int(model.predict(x_sample)[0])
            benign_probability, ransomware_probability = get_class_probabilities(model, x_sample)
            confidence = max(benign_probability, ransomware_probability)
            st.metric("Actual Label (Ground Truth)", label_to_text(actual_label))

    with live_mid:
        with st.container(border=True):
            section_heading("Prediction Confidence")
            render_confidence_ring(confidence)

    with live_right:
        with st.container(border=True):
            section_heading("Predicted Label")
            result_class = "bad" if prediction == 1 else ""
            render_html(f'<div class="result-pill {result_class}">{"🔴" if prediction == 1 else "🟢"} {label_to_text(prediction)}</div>')
            section_heading("Prediction Probabilities")
            render_probability_rows(benign_probability, ransomware_probability)

    render_html('<div id="model-performance"></div>')
    performance_col, matrix_col = st.columns([1.25, 1], gap="medium")

    with performance_col:
        with st.container(border=True):
            section_heading("Model Performance (Test Set)", "Metrics are calculated from the saved model on the MLRan test data.")
            render_metric_grid(metrics)

    with matrix_col:
        render_html('<div id="confusion-matrix"></div>')
        with st.container(border=True):
            section_heading("Confusion Matrix (Test Set)", "Actual labels compared with model predictions.")
            render_confusion_matrix(metrics)

    render_html('<div id="feature-analysis"></div>')
    features_col, sample_col, reasons_col = st.columns([1.1, 1.05, 1.15], gap="medium")

    with features_col:
        with st.container(border=True):
            section_heading("Top Important Features", "Top coefficient-based behavioural features from the saved model.")
            render_feature_bars(importance_df)

    with sample_col:
        with st.container(border=True):
            section_heading("Selected Sample – Behavioural Features", "Preview of high-weight feature values for the selected sample.")
            sample_preview_df = build_sample_feature_preview(importance_df, selected_row)
            if not sample_preview_df.empty:
                st.dataframe(sample_preview_df, width="stretch", hide_index=True)
            st.caption(f"Total features in selected sample: {len(feature_columns)}")
            with st.expander("Show all selected sample metadata and features"):
                metadata_df = selected_row[["sample_id", "sample_type", "family_label", "type_label"]].to_frame("Value")
                metadata_df["Value"] = metadata_df["Value"].astype(str)
                st.dataframe(metadata_df, width="stretch", hide_index=False)
                st.dataframe(x_sample, width="stretch")

    with reasons_col:
        with st.container(border=True):
            section_heading("Why This Prediction?", "Thesis-safe interpretation based on available model outputs.")
            render_prediction_reasons(prediction, confidence, importance_df, selected_row)

    render_html('<div id="how-it-works"></div>')
    bottom_left, bottom_right = st.columns([2.6, 0.72], gap="medium")

    with bottom_left:
        with st.container(border=True):
            section_heading("How It Works")
            render_workflow()

    with bottom_right:
        render_html('<div id="about-project"></div>')
        with st.container(border=True):
            section_heading("Project Information")
            render_html(
                """
                <ul class="info-list">
                    <li><strong>Project Type:</strong> MSc Dissertation</li>
                    <li><strong>Domain:</strong> Cyber Security</li>
                    <li><strong>Institution:</strong> Arden University</li>
                    <li><strong>Developer:</strong> Keshav Vasalli</li>
                    <li><strong>Year:</strong> 2026</li>
                </ul>
                """
            )

except FileNotFoundError as error:
    st.error("Required project file was not found.")
    st.write(error)
    st.warning("Check that the saved model, feature column list, and local MLRan test dataset are present in the expected folders.")
except KeyError as error:
    st.error("A required column was not found in the dataset.")
    st.write(error)
    st.warning("Check that the MLRan test CSV still contains sample_id, sample_type, family_label, type_label, and all saved feature columns.")
except Exception as error:
    st.error("The dashboard could not be loaded.")
    st.write(error)
