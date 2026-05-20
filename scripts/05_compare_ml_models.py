from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "MLRan_X_train_RFE.csv"
X_TEST_PATH = DATA_DIR / "MLRan_X_test_RFE.csv"

REPORT_PATH = REPORTS_DIR / "model_comparison_report.txt"
CSV_PATH = REPORTS_DIR / "model_comparison_metrics.csv"


def evaluate_model(model_name, model, x_train, y_train, x_test, y_test):
    print(f"Training {model_name}...")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["goodware_benign", "ransomware"],
        zero_division=0,
    )

    return metrics, cm, report


def main():
    print("Loading MLRan training and testing data...")

    train_df = pd.read_csv(X_TRAIN_PATH)
    test_df = pd.read_csv(X_TEST_PATH)

    non_feature_columns = ["sample_id", "sample_type", "family_label", "type_label"]

    x_train = train_df.drop(columns=non_feature_columns)
    y_train = train_df["sample_type"]

    x_test = test_df.drop(columns=non_feature_columns)
    y_test = test_df["sample_type"]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Logistic Regression": Pipeline(
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
        ),
    }

    all_metrics = []
    report_lines = []

    report_lines.append("ML Model Comparison Report")
    report_lines.append("=" * 45)
    report_lines.append("")
    report_lines.append("Task: Binary ransomware detection")
    report_lines.append("Target column: sample_type")
    report_lines.append("Class 0: goodware / benign")
    report_lines.append("Class 1: ransomware")
    report_lines.append("")
    report_lines.append(f"Training shape: {x_train.shape}")
    report_lines.append(f"Testing shape: {x_test.shape}")
    report_lines.append("")

    for model_name, model in models.items():
        metrics, cm, class_report = evaluate_model(
            model_name, model, x_train, y_train, x_test, y_test
        )

        all_metrics.append(metrics)

        report_lines.append(model_name)
        report_lines.append("-" * 45)
        report_lines.append(f"Accuracy:  {metrics['accuracy']:.4f}")
        report_lines.append(f"Precision: {metrics['precision']:.4f}")
        report_lines.append(f"Recall:    {metrics['recall']:.4f}")
        report_lines.append(f"F1-score:  {metrics['f1_score']:.4f}")
        report_lines.append("")
        report_lines.append("Confusion Matrix:")
        report_lines.append(str(cm))
        report_lines.append("")
        report_lines.append("Classification Report:")
        report_lines.append(class_report)
        report_lines.append("")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(CSV_PATH, index=False)

    best_model = metrics_df.sort_values(by="f1_score", ascending=False).iloc[0]

    report_lines.append("Best Model Based on F1-score")
    report_lines.append("-" * 45)
    report_lines.append(f"Model: {best_model['model']}")
    report_lines.append(f"F1-score: {best_model['f1_score']:.4f}")
    report_lines.append("")
    report_lines.append("Safety Note")
    report_lines.append("-" * 45)
    report_lines.append("This comparison uses processed behavioural feature data only.")
    report_lines.append("No live ransomware binaries are downloaded, executed, or handled.")

    final_report = "\n".join(report_lines)

    REPORT_PATH.write_text(final_report, encoding="utf-8")

    print(final_report)
    print(f"\nModel comparison report saved to: {REPORT_PATH}")
    print(f"Model comparison CSV saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()