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
    page_icon="🛡️",
    layout="wide",
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
    return pd.read_csv(TEST_DATA_PATH)


def label_to_text(label: int) -> str:
    if label == 0:
        return "Goodware / Benign"
    if label == 1:
        return "Ransomware"
    return "Unknown"


st.title("AI-Driven Behavioral Ransomware Detection System")
st.write(
    "This prototype uses a trained Logistic Regression model to classify "
    "processed behavioural feature data as either benign goodware or ransomware."
)

st.info(
    "Safety note: This app uses already extracted behavioural features only. "
    "No live ransomware files are uploaded, executed, or handled."
)

try:
    model = load_model()
    feature_columns = load_feature_columns()
    test_df = load_test_data()

    st.success("Saved model, feature columns, and MLRan test dataset loaded successfully.")

    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "Logistic Regression")

    with col2:
        st.metric("Feature Count", len(feature_columns))

    with col3:
        st.metric("Test Samples", len(test_df))

    st.subheader("Select a Test Sample")

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

    st.subheader("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Actual Label", label_to_text(actual_label))

    with result_col2:
        st.metric("Predicted Label", label_to_text(prediction))

    with result_col3:
        confidence = max(benign_probability, ransomware_probability)
        st.metric("Prediction Confidence", f"{confidence:.2%}")

    if prediction == 1:
        st.error("Prediction: Ransomware behaviour detected.")
    else:
        st.success("Prediction: Goodware / benign behaviour detected.")

    st.subheader("Class Probability Scores")

    probability_df = pd.DataFrame(
        {
            "Class": ["Goodware / Benign", "Ransomware"],
            "Probability": [benign_probability, ransomware_probability],
        }
    )

    st.dataframe(probability_df, width="stretch")
    st.bar_chart(probability_df.set_index("Class"))

    with st.expander("Show selected sample metadata"):
        st.write(
            selected_row[
                ["sample_id", "sample_type", "family_label", "type_label"]
            ].to_frame("value")
        )

    with st.expander("Show selected behavioural feature values"):
        st.dataframe(X_sample, width="stretch")

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