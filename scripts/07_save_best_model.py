from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "raw" / "mlran"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "MLRan_X_train_RFE.csv"
X_TEST_PATH = DATA_DIR / "MLRan_X_test_RFE.csv"

MODEL_PATH = MODELS_DIR / "best_logistic_regression_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
REPORT_PATH = REPORTS_DIR / "best_model_report.txt"


def main():
    print("Loading MLRan training and testing data...")

    train_df = pd.read_csv(X_TRAIN_PATH)
    test_df = pd.read_csv(X_TEST_PATH)

    non_feature_columns = ["sample_id", "sample_type", "family_label", "type_label"]

    X_train = train_df.drop(columns=non_feature_columns)
    y_train = train_df["sample_type"]

    X_test = test_df.drop(columns=non_feature_columns)
    y_test = test_df["sample_type"]

    print("Training best model: Logistic Regression...")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    print("Evaluating best model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(
        y_test,
        y_pred,
        target_names=["goodware_benign", "ransomware"],
        zero_division=0,
    )

    print("Saving trained model and feature columns...")

    joblib.dump(model, MODEL_PATH)

    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as file:
        json.dump(list(X_train.columns), file, indent=4)

    report_lines = []
    report_lines.append("Best Model Training Report")
    report_lines.append("=" * 45)
    report_lines.append("")
    report_lines.append("Selected model: Logistic Regression")
    report_lines.append("Reason: Best F1-score during model comparison")
    report_lines.append("")
    report_lines.append("Task: Binary ransomware detection")
    report_lines.append("Target column: sample_type")
    report_lines.append("Class 0: goodware / benign")
    report_lines.append("Class 1: ransomware")
    report_lines.append("")
    report_lines.append(f"Training shape: {X_train.shape}")
    report_lines.append(f"Testing shape: {X_test.shape}")
    report_lines.append("")
    report_lines.append("Evaluation Metrics")
    report_lines.append("-" * 45)
    report_lines.append(f"Accuracy:  {accuracy:.4f}")
    report_lines.append(f"Precision: {precision:.4f}")
    report_lines.append(f"Recall:    {recall:.4f}")
    report_lines.append(f"F1-score:  {f1:.4f}")
    report_lines.append("")
    report_lines.append("Confusion Matrix")
    report_lines.append("-" * 45)
    report_lines.append(str(cm))
    report_lines.append("")
    report_lines.append("Classification Report")
    report_lines.append("-" * 45)
    report_lines.append(class_report)
    report_lines.append("")
    report_lines.append("Saved Files")
    report_lines.append("-" * 45)
    report_lines.append(f"Model file: {MODEL_PATH}")
    report_lines.append(f"Feature columns file: {FEATURE_COLUMNS_PATH}")
    report_lines.append("")
    report_lines.append("Safety Note")
    report_lines.append("-" * 45)
    report_lines.append("This model uses processed behavioural feature data only.")
    report_lines.append("No live ransomware binaries are downloaded, executed, or handled.")

    report_text = "\n".join(report_lines)

    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nBest model saved to: {MODEL_PATH}")
    print(f"Feature columns saved to: {FEATURE_COLUMNS_PATH}")
    print(f"Best model report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()