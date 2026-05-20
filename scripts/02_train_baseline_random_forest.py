from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "raw" / "mlran"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "MLRan_X_train_RFE.csv"
X_TEST_PATH = DATA_DIR / "MLRan_X_test_RFE.csv"

REPORT_PATH = REPORTS_DIR / "baseline_random_forest_report.txt"


def main():
    print("Loading MLRan training and testing data...")

    train_df = pd.read_csv(X_TRAIN_PATH)
    test_df = pd.read_csv(X_TEST_PATH)

    non_feature_columns = ["sample_id", "sample_type", "family_label", "type_label"]

    X_train = train_df.drop(columns=non_feature_columns)
    y_train = train_df["sample_type"]

    X_test = test_df.drop(columns=non_feature_columns)
    y_test = test_df["sample_type"]

    print("Training Random Forest baseline model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["goodware_benign", "ransomware"],
        zero_division=0,
    )

    output = []
    output.append("Baseline Random Forest Model Report")
    output.append("=" * 45)
    output.append("")
    output.append("Task: Binary ransomware detection")
    output.append("Target column: sample_type")
    output.append("Class 0: goodware / benign")
    output.append("Class 1: ransomware")
    output.append("")
    output.append(f"Training shape: {X_train.shape}")
    output.append(f"Testing shape: {X_test.shape}")
    output.append("")
    output.append("Model: RandomForestClassifier")
    output.append("n_estimators: 100")
    output.append("random_state: 42")
    output.append("class_weight: balanced")
    output.append("")
    output.append("Evaluation Metrics")
    output.append("-" * 45)
    output.append(f"Accuracy:  {accuracy:.4f}")
    output.append(f"Precision: {precision:.4f}")
    output.append(f"Recall:    {recall:.4f}")
    output.append(f"F1-score:  {f1:.4f}")
    output.append("")
    output.append("Confusion Matrix")
    output.append("-" * 45)
    output.append(str(cm))
    output.append("")
    output.append("Classification Report")
    output.append("-" * 45)
    output.append(report)
    output.append("")
    output.append("Safety Note")
    output.append("-" * 45)
    output.append("This model uses processed behavioural feature data only.")
    output.append("No live ransomware binaries are downloaded, executed, or handled.")

    output_text = "\n".join(output)

    print(output_text)

    REPORT_PATH.write_text(output_text, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()