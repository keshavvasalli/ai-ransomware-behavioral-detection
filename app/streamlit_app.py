from pathlib import Path
import json
from html import escape

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
        --bg-panel-soft: rgba(11, 27, 49, 0.70);
        --cyan: #00E5FF;
        --cyan-soft: rgba(0, 229, 255, 0.12);
        --blue: #2F80ED;
        --violet: #8B5CF6;
        --green: #22C55E;
        --red: #FF3B6B;
        --amber: #F59E0B;
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
        padding-top: 1rem !important;
        padding-right: 1.05rem !important;
        padding-bottom: 1.2rem !important;
        padding-left: 17.2rem !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.72rem;
    }

    .fixed-sidebar {
        position: fixed;
        left: 0.75rem;
        top: 0.75rem;
        bottom: 0.75rem;
        width: 14.9rem;
        z-index: 999;
        border: 1px solid rgba(0, 229, 255, 0.20);
        border-radius: 20px;
        background:
            radial-gradient(circle at 28% 8%, rgba(0, 229, 255, 0.16), transparent 25%),
            linear-gradient(180deg, rgba(7, 16, 31, 0.98), rgba(2, 6, 17, 0.98));
        box-shadow: 0 22px 70px rgba(0, 0, 0, 0.40), var(--shadow-glow);
        padding: 1.25rem 1.05rem;
        overflow: hidden;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        margin-bottom: 1.25rem;
    }

    .brand-shield {
        width: 2.9rem;
        height: 2.9rem;
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
        margin: 0.38rem 0;
        padding: 0.72rem 0.74rem;
        color: var(--text-soft);
        font-size: 0.83rem;
        border-radius: 13px;
        border: 1px solid transparent;
    }

    .nav-item.active {
        background: linear-gradient(90deg, rgba(47, 128, 237, 0.38), rgba(0, 229, 255, 0.09));
        border-color: rgba(0, 229, 255, 0.26);
        box-shadow: inset -3px 0 0 rgba(0, 229, 255, 0.86);
    }

    .nav-icon {
        width: 1.12rem;
        text-align: center;
        color: var(--cyan);
    }

    .sidebar-card {
        position: absolute;
        left: 1.05rem;
        right: 1.05rem;
        bottom: 5.35rem;
        border: 1px solid rgba(0, 229, 255, 0.20);
        border-radius: 16px;
        background: rgba(8, 18, 34, 0.72);
        padding: 0.9rem;
        color: var(--text-soft);
        font-size: 0.76rem;
        line-height: 1.55;
    }

    .sidebar-stack {
        position: absolute;
        left: 1.05rem;
        right: 1.05rem;
        bottom: 1.1rem;
        border-top: 1px solid rgba(0, 229, 255, 0.14);
        padding-top: 0.75rem;
        color: var(--text-muted);
        font-size: 0.73rem;
        line-height: 1.45;
    }

    .dash-card {
        border: 1px solid var(--border);
        border-radius: 18px;
        background:
            linear-gradient(145deg, rgba(8, 18, 34, 0.90), rgba(12, 26, 48, 0.64)),
            radial-gradient(circle at 90% 0%, rgba(0, 229, 255, 0.08), transparent 30%);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28), var(--shadow-glow);
        padding: 1rem;
        height: 100%;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.65fr) minmax(260px, 0.95fr) minmax(250px, 0.85fr);
        gap: 0.85rem;
        margin-bottom: 0.85rem;
    }

    .hero-panel {
        min-height: 250px;
        border: 1px solid rgba(0, 229, 255, 0.24);
        border-radius: 20px;
        background:
            linear-gradient(135deg, rgba(8, 18, 34, 0.94), rgba(11, 22, 45, 0.74)),
            radial-gradient(circle at 95% 18%, rgba(139, 92, 246, 0.18), transparent 34%);
        box-shadow: 0 22px 70px rgba(0, 0, 0, 0.32), var(--shadow-glow);
        padding: 1.45rem;
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
        font-size: clamp(2rem, 3.15vw, 3.45rem);
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
        font-size: 0.96rem;
        line-height: 1.55;
        margin-top: 0.85rem;
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

    .chip-label {
        color: var(--text-muted);
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.22rem;
    }

    .chip-value {
        color: var(--text-main);
        font-weight: 820;
        font-size: 0.88rem;
    }

    .safety-note {
        margin-top: 1rem;
        border: 1px solid rgba(0, 229, 255, 0.21);
        background: rgba(0, 229, 255, 0.055);
        border-radius: 13px;
        padding: 0.72rem 0.86rem;
        color: #DDF8FF;
        font-size: 0.78rem;
        line-height: 1.45;
    }

    .visual-card {
        min-height: 250px;
        display: grid;
        place-items: center;
        border: 1px solid rgba(0, 229, 255, 0.18);
        border-radius: 20px;
        background:
            radial-gradient(circle at 50% 50%, rgba(0, 229, 255, 0.18), transparent 24%),
            radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.16), transparent 45%),
            rgba(8, 18, 34, 0.56);
        overflow: hidden;
        position: relative;
    }

    .orb {
        position: relative;
        width: 210px;
        height: 210px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background:
            repeating-radial-gradient(circle, transparent 0 16px, rgba(0, 229, 255, 0.07) 17px 18px),
            radial-gradient(circle, rgba(0, 229, 255, 0.16), rgba(139, 92, 246, 0.08) 45%, transparent 70%);
        border: 1px solid rgba(0, 229, 255, 0.24);
        box-shadow: 0 0 52px rgba(0, 229, 255, 0.22), inset 0 0 42px rgba(139, 92, 246, 0.16);
    }

    .orb:before,
    .orb:after {
        content: "";
        position: absolute;
        inset: 22px;
        border: 1px solid rgba(0, 229, 255, 0.22);
        border-radius: 50%;
        transform: rotate(24deg);
    }

    .orb:after {
        inset: 44px;
        border-color: rgba(139, 92, 246, 0.30);
        transform: rotate(-32deg);
    }

    .shield-core {
        width: 78px;
        height: 94px;
        display: grid;
        place-items: center;
        color: #EAFBFF;
        font-size: 2.1rem;
        border-radius: 34px 34px 44px 44px;
        background:
            linear-gradient(145deg, rgba(0, 229, 255, 0.25), rgba(139, 92, 246, 0.16)),
            rgba(8, 18, 34, 0.78);
        border: 1px solid rgba(0, 229, 255, 0.36);
        box-shadow: 0 0 34px rgba(0, 229, 255, 0.30);
        z-index: 2;
    }

    .system-panel {
        min-height: 250px;
        border: 1px solid rgba(0, 229, 255, 0.20);
        border-radius: 20px;
        background: rgba(8, 18, 34, 0.72);
        padding: 1rem;
    }

    .panel-title {
        color: var(--text-main);
        font-size: 0.98rem;
        font-weight: 900;
        margin-bottom: 0.75rem;
    }

    .sys-row {
        display: grid;
        grid-template-columns: 1.8rem 1fr;
        gap: 0.55rem;
        align-items: center;
        margin-bottom: 0.72rem;
    }

    .sys-icon {
        width: 1.72rem;
        height: 1.72rem;
        display: grid;
        place-items: center;
        border-radius: 9px;
        background: rgba(47, 128, 237, 0.13);
        color: var(--cyan);
        border: 1px solid rgba(0, 229, 255, 0.16);
    }

    .sys-label {
        color: var(--text-muted);
        font-size: 0.66rem;
        line-height: 1.15;
    }

    .sys-value {
        color: var(--text-main);
        font-size: 0.82rem;
        font-weight: 820;
        line-height: 1.18;
    }

    .section-head {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 1rem;
        margin: 0.2rem 0 0.6rem 0;
    }

    .section-title {
        color: var(--text-main);
        font-size: 1.06rem;
        font-weight: 920;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .section-sub {
        color: var(--text-muted);
        font-size: 0.75rem;
        margin-top: 0.15rem;
    }

    .result-pill {
        border-radius: 13px;
        padding: 0.78rem 0.92rem;
        font-size: 0.9rem;
        font-weight: 900;
        border: 1px solid rgba(34, 197, 94, 0.32);
        background: rgba(34, 197, 94, 0.12);
        color: #B9FBCB;
    }

    .result-pill.bad {
        border-color: rgba(255, 59, 107, 0.38);
        background: rgba(255, 59, 107, 0.12);
        color: #FFD1DC;
    }

    .confidence-ring {
        width: 154px;
        height: 154px;
        margin: 0.1rem auto 0.6rem auto;
        border-radius: 50%;
        background:
            radial-gradient(circle at center, #06101F 0 52%, transparent 53%),
            conic-gradient(var(--green) var(--pct), rgba(0, 229, 255, 0.18) 0);
        display: grid;
        place-items: center;
        box-shadow: 0 0 32px rgba(34, 197, 94, 0.18);
    }

    .ring-inner {
        width: 114px;
        height: 114px;
        border-radius: 50%;
        background: #030711;
        display: grid;
        place-items: center;
        border: 1px solid rgba(0, 229, 255, 0.12);
    }

    .ring-value {
        color: var(--text-main);
        font-size: 1.75rem;
        font-weight: 950;
        letter-spacing: -0.04em;
    }

    .ring-label {
        color: var(--green);
        font-size: 0.68rem;
        font-weight: 800;
        text-align: center;
        margin-top: -0.8rem;
    }

    .prob-row {
        display: grid;
        grid-template-columns: 1.25fr 0.65fr 2fr;
        gap: 0.55rem;
        align-items: center;
        margin: 0.7rem 0;
        color: var(--text-soft);
        font-size: 0.82rem;
    }

    .bar-track {
        height: 0.52rem;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.13);
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--cyan), var(--green));
    }

    .bar-fill.red {
        background: linear-gradient(90deg, var(--red), var(--amber));
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
    }

    .metric-card {
        border: 1px solid rgba(0, 229, 255, 0.17);
        background:
            radial-gradient(circle at 92% 8%, rgba(47, 128, 237, 0.20), transparent 34%),
            rgba(6, 14, 31, 0.58);
        border-radius: 15px;
        padding: 0.86rem;
        min-height: 6.2rem;
    }

    .metric-name {
        color: var(--cyan);
        font-size: 0.72rem;
        margin-bottom: 0.45rem;
    }

    .metric-number {
        color: var(--text-main);
        font-size: 1.35rem;
        font-weight: 950;
    }

    .spark {
        height: 28px;
        margin-top: 0.48rem;
        border-radius: 12px;
        background:
            linear-gradient(135deg, transparent 20%, rgba(0, 229, 255, 0.20) 21%, transparent 22%),
            linear-gradient(160deg, transparent 35%, rgba(47, 128, 237, 0.28) 36%, transparent 37%),
            rgba(0, 229, 255, 0.04);
        border: 1px solid rgba(0, 229, 255, 0.07);
    }

    .cmatrix {
        width: 100%;
        border-collapse: collapse;
        color: var(--text-soft);
        font-size: 0.78rem;
        overflow: hidden;
        border-radius: 12px;
    }

    .cmatrix th,
    .cmatrix td {
        border: 1px solid rgba(0, 229, 255, 0.12);
        padding: 0.54rem 0.46rem;
        text-align: center;
    }

    .cmatrix th {
        color: var(--text-main);
        background: rgba(47, 128, 237, 0.15);
        font-weight: 850;
    }

    .cmatrix td {
        background: rgba(6, 14, 31, 0.48);
    }

    .feature-row {
        display: grid;
        grid-template-columns: 1.5fr 2.2fr 0.58fr;
        gap: 0.56rem;
        align-items: center;
        margin: 0.47rem 0;
        font-size: 0.72rem;
        color: var(--text-soft);
    }

    .feature-name {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .feature-track {
        height: 0.44rem;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.13);
        overflow: hidden;
    }

    .feature-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #536DFE, var(--violet), var(--cyan));
    }

    .feature-value {
        text-align: right;
        color: var(--text-muted);
    }

    .reason-list {
        margin: 0.3rem 0 0 0;
        padding: 0;
        list-style: none;
    }

    .reason-list li {
        display: grid;
        grid-template-columns: 1.1rem 1fr;
        gap: 0.52rem;
        margin: 0.58rem 0;
        color: var(--text-soft);
        font-size: 0.78rem;
        line-height: 1.42;
    }

    .check {
        color: var(--green);
        font-weight: 950;
    }

    .flow-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.75rem;
    }

    .flow-card {
        min-height: 5.7rem;
        border: 1px solid rgba(0, 229, 255, 0.16);
        border-radius: 14px;
        background: rgba(6, 14, 31, 0.52);
        padding: 0.78rem;
    }

    .flow-number {
        color: var(--cyan);
        font-size: 0.72rem;
        font-weight: 900;
        margin-bottom: 0.28rem;
    }

    .flow-title {
        color: var(--text-main);
        font-size: 0.78rem;
        font-weight: 850;
        margin-bottom: 0.18rem;
    }

    .flow-copy {
        color: var(--text-muted);
        font-size: 0.68rem;
        line-height: 1.35;
    }

    .info-list {
        color: var(--text-soft);
        font-size: 0.76rem;
        line-height: 1.55;
        margin: 0.2rem 0 0 0.9rem;
        padding: 0;
    }

    .info-list li {
        margin: 0.22rem 0;
    }

    .stSelectbox label,
    .stDataFrame,
    div[data-testid="stMetric"] {
        color: var(--text-main) !important;
    }

    div[data-baseweb="select"] > div {
        background: rgba(6, 14, 31, 0.76) !important;
        border-color: rgba(0, 229, 255, 0.18) !important;
        color: var(--text-main) !important;
        border-radius: 12px !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(6, 14, 31, 0.54);
        border: 1px solid rgba(0, 229, 255, 0.14);
        border-radius: 14px;
        padding: 0.68rem 0.76rem;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text-main);
        font-weight: 900;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
    }

    .stDataFrame {
        border: 1px solid rgba(0, 229, 255, 0.16);
        border-radius: 12px;
        overflow: hidden;
    }

    @media (max-width: 1150px) {
        .block-container {
            padding-left: 1rem !important;
        }

        .fixed-sidebar {
            display: none;
        }

        .hero-grid {
            grid-template-columns: 1fr;
        }

        .metric-grid,
        .flow-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
</style>
"""


def render_sidebar():
    st.markdown(
        """
        <div class="fixed-sidebar">
            <div class="brand-row">
                <div class="brand-shield">AI</div>
                <div>
                    <div class="brand-title">AI-Driven<br>Behavioral</div>
                    <div class="brand-subtitle">Ransomware Detection System</div>
                </div>
            </div>

            <div class="nav-item active"><span class="nav-icon">⌂</span><span>Dashboard</span></div>
            <div class="nav-item"><span class="nav-icon">◌</span><span>Live Detection</span></div>
            <div class="nav-item"><span class="nav-icon">⌁</span><span>Model Performance</span></div>
            <div class="nav-item"><span class="nav-icon">▦</span><span>Confusion Matrix</span></div>
            <div class="nav-item"><span class="nav-icon">⌬</span><span>Feature Analysis</span></div>
            <div class="nav-item"><span class="nav-icon">◇</span><span>How It Works</span></div>
            <div class="nav-item"><span class="nav-icon">ⓘ</span><span>About Project</span></div>

            <div class="sidebar-card">
                <strong>About This Project</strong><br><br>
                Research prototype developed as part of the MSc Cyber Security dissertation at Arden University.
            </div>

            <div class="sidebar-stack">
                <strong>Tech Stack</strong><br>
                Python · Scikit-learn · Streamlit<br>
                Pandas · Joblib · MLRan<br><br>
                © 2026 | MSc Cyber Security<br>
                Arden University
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def label_to_text(label: int) -> str:
    if label == 0:
        return "Benign (Goodware)"
    if label == 1:
        return "Ransomware"
    return "Unknown"


def label_to_short(label: int) -> str:
    if label == 0:
        return "Benign"
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

    if hasattr(model_object, "classes_"):
        classes = list(model_object.classes_)
    else:
        estimator = extract_final_estimator(model_object)
        classes = list(getattr(estimator, "classes_", [0, 1]))

    benign_probability = probabilities[classes.index(0)] if 0 in classes else probabilities[0]
    ransomware_probability = probabilities[classes.index(1)] if 1 in classes else probabilities[1]

    return float(benign_probability), float(ransomware_probability)


def render_hero(total_features: int, total_samples: int):
    st.markdown(
        f"""
        <div class="hero-grid">
            <div class="hero-panel">
                <div class="eyebrow">Research Prototype</div>
                <h1 class="hero-title">AI-Driven Behavioral<br><span>Ransomware Detection System</span></h1>
                <div class="hero-subtitle">
                    Machine learning based classification of files as benign/goodware or ransomware using
                    processed behavioural feature data from the MLRan dataset.
                </div>
                <div class="hero-mini-grid">
                    <div class="mini-chip">
                        <div class="chip-label">Dataset</div>
                        <div class="chip-value">MLRan (Processed)</div>
                    </div>
                    <div class="mini-chip">
                        <div class="chip-label">Best Model</div>
                        <div class="chip-value">Logistic Regression</div>
                    </div>
                </div>
                <div class="safety-note">
                    <strong>Safety Note:</strong> This dashboard uses pre-extracted behavioural features only.
                    No live ransomware files are uploaded, executed, or analysed.
                </div>
            </div>

            <div class="visual-card">
                <div class="orb">
                    <div class="shield-core">🔒</div>
                </div>
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
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, subtitle: str = ""):
    subtitle_html = f'<div class="section-sub">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-head">
            <div>
                <h2 class="section-title">{escape(title)}</h2>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_ring(confidence: float):
    confidence_pct = max(0, min(confidence * 100, 100))
    confidence_label = "High Confidence" if confidence >= 0.80 else "Moderate Confidence"
    st.markdown(
        f"""
        <div class="confidence-ring" style="--pct: {confidence_pct:.2f}%;">
            <div class="ring-inner">
                <div>
                    <div class="ring-value">{confidence_pct:.1f}%</div>
                    <div class="ring-label">{confidence_label}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_rows(benign_probability: float, ransomware_probability: float):
    benign_width = benign_probability * 100
    ransomware_width = ransomware_probability * 100
    st.markdown(
        f"""
        <div class="prob-row">
            <div>🟢 Benign</div>
            <div>{benign_width:.1f}%</div>
            <div class="bar-track"><div class="bar-fill" style="width: {benign_width:.2f}%"></div></div>
        </div>
        <div class="prob-row">
            <div>🔴 Ransomware</div>
            <div>{ransomware_width:.1f}%</div>
            <div class="bar-track"><div class="bar-fill red" style="width: {ransomware_width:.2f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics):
    metric_items = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1-score", metrics["f1"]),
    ]

    cards = ""
    for label, value in metric_items:
        cards += f"""
        <div class="metric-card">
            <div class="metric-name">{escape(label)}</div>
            <div class="metric-number">{percentage(value)}</div>
            <div class="spark"></div>
        </div>
        """

    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def render_confusion_matrix(metrics):
    tn = metrics["tn"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    tp = metrics["tp"]
    benign_total = tn + fp
    ransomware_total = fn + tp
    pred_benign_total = tn + fn
    pred_ransomware_total = fp + tp
    total = benign_total + ransomware_total

    st.markdown(
        f"""
        <table class="cmatrix">
            <tr>
                <th>Actual \\ Predicted</th>
                <th>Benign</th>
                <th>Ransomware</th>
                <th>Total</th>
            </tr>
            <tr>
                <th>Benign</th>
                <td>{tn}</td>
                <td>{fp}</td>
                <td>{benign_total}</td>
            </tr>
            <tr>
                <th>Ransomware</th>
                <td>{fn}</td>
                <td>{tp}</td>
                <td>{ransomware_total}</td>
            </tr>
            <tr>
                <th>Total</th>
                <td>{pred_benign_total}</td>
                <td>{pred_ransomware_total}</td>
                <td>{total}</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_feature_bars(importance_df):
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
        rows_html += f"""
        <div class="feature-row">
            <div class="feature-name" title="{feature}">{feature}</div>
            <div class="feature-track"><div class="feature-fill" style="width: {width:.2f}%"></div></div>
            <div class="feature-value">{coeff:.3f}</div>
        </div>
        """

    st.markdown(rows_html, unsafe_allow_html=True)


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


def render_prediction_reasons(prediction: int, confidence: float, importance_df, selected_row):
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

    list_items = "".join(
        f'<li><span class="check">✓</span><span>{escape(reason)}</span></li>'
        for reason in reasons
    )

    st.markdown(f'<ul class="reason-list">{list_items}</ul>', unsafe_allow_html=True)


def render_workflow():
    workflow_items = [
        ("1. Data Collection", "Behavioural features extracted from files in a controlled dataset."),
        ("2. Preprocessing", "Feature-selected data prepared for model training and testing."),
        ("3. Model Training", "Machine learning models compared, with Logistic Regression selected."),
        ("4. Prediction", "Selected test samples classified as benign or ransomware."),
        ("5. Analysis", "Confidence, metrics and feature signals presented for interpretation."),
    ]

    cards = ""
    for title, description in workflow_items:
        cards += f"""
        <div class="flow-card">
            <div class="flow-number">{escape(title.split('.')[0])}</div>
            <div class="flow-title">{escape(title.split('. ', 1)[1])}</div>
            <div class="flow-copy">{escape(description)}</div>
        </div>
        """

    st.markdown(f'<div class="flow-grid">{cards}</div>', unsafe_allow_html=True)


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
render_sidebar()


try:
    model = load_model()
    feature_columns = load_feature_columns()
    test_df = load_test_data()

    metrics = calculate_metrics(model, test_df, feature_columns)
    importance_df = get_feature_importance(model, feature_columns)

    render_hero(total_features=len(feature_columns), total_samples=len(test_df))

    live_left, live_mid, live_right = st.columns([1.35, 0.92, 1.65], gap="medium")

    with live_left:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

    with live_mid:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        section_heading("Prediction Confidence")
        render_confidence_ring(confidence)
        st.markdown("</div>", unsafe_allow_html=True)

    with live_right:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        section_heading("Predicted Label")
        result_class = "bad" if prediction == 1 else ""
        st.markdown(
            f'<div class="result-pill {result_class}">{"🔴" if prediction == 1 else "🟢"} {label_to_text(prediction)}</div>',
            unsafe_allow_html=True,
        )
        section_heading("Prediction Probabilities")
        render_probability_rows(benign_probability, ransomware_probability)
        st.markdown("</div>", unsafe_allow_html=True)

    performance_col, matrix_col = st.columns([1.25, 1], gap="medium")

    with performance_col:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        section_heading("Model Performance (Test Set)", "Metrics are calculated from the saved model on the MLRan test data.")
        render_metric_grid(metrics)
        st.markdown("</div>", unsafe_allow_html=True)

    with matrix_col:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        section_heading("Confusion Matrix (Test Set)", "Actual labels compared with model predictions.")
        render_confusion_matrix(metrics)
        st.markdown("</div>", unsafe_allow_html=True)

    features_col, sample_col, reasons_col = st.columns([1.1, 1.05, 1.15], gap="medium")

    with features_col:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        section_heading("Top Important Features", "Top coefficient-based behavioural features from the saved model.")
        render_feature_bars(importance_df)
        st.markdown("</div>", unsafe_allow_html=True)

    with sample_col:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        section_heading("Selected Sample – Behavioural Features", "Preview of high-weight feature values for the selected sample.")
        sample_preview_df = build_sample_feature_preview(importance_df, selected_row)
        if not sample_preview_df.empty:
            st.dataframe(sample_preview_df, width="stretch", hide_index=True)
        st.caption(f"Total features in selected sample: {len(feature_columns)}")
        with st.expander("Show all selected sample metadata and features"):
            metadata_df = selected_row[
                ["sample_id", "sample_type", "family_label", "type_label"]
            ].to_frame("Value")
            st.dataframe(metadata_df, width="stretch")
            st.dataframe(x_sample, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with reasons_col:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        section_heading("Why This Prediction?", "Thesis-safe interpretation based on available model outputs.")
        render_prediction_reasons(prediction, confidence, importance_df, selected_row)
        st.markdown("</div>", unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([2.6, 0.72], gap="medium")

    with bottom_left:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        section_heading("How It Works")
        render_workflow()
        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_right:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        section_heading("Project Information")
        st.markdown(
            """
            <ul class="info-list">
                <li><strong>Project Type:</strong> MSc Dissertation</li>
                <li><strong>Domain:</strong> Cyber Security</li>
                <li><strong>Institution:</strong> Arden University</li>
                <li><strong>Developer:</strong> Keshav Vasalli</li>
                <li><strong>Year:</strong> 2026</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

except FileNotFoundError as error:
    st.error("Required project file was not found.")
    st.write(error)
    st.warning(
        "Check that the saved model, feature column list, and local MLRan test dataset are present in the expected folders."
    )
except KeyError as error:
    st.error("A required column was not found in the dataset.")
    st.write(error)
    st.warning("Check that the MLRan test CSV still contains sample_id, sample_type, and all saved feature columns.")
except Exception as error:
    st.error("The dashboard could not be loaded.")
    st.write(error)
