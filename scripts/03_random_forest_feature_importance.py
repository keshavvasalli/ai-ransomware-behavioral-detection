from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "raw" / "mlran"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "MLRan_X_train_RFE.csv"
X_TEST_PATH = DATA_DIR / "MLRan_X_test_RFE.csv"
FEATURE_NAMES_PATH = DATA_DIR / "RFE_selected_feature_names_dic.json"

CSV_OUTPUT_PATH = REPORTS_DIR / "random_forest_feature_importance.csv"
TXT_OUTPUT_PATH = REPORTS_DIR / "random_forest_feature_importance_report.txt"


def get_feature_category(feature_name: str) -> str:
    if ":" in feature_name:
        return feature_name.split(":", 1)[0]
    return "UNKNOWN"


def main():
    print("Loading MLRan dataset files...")

    train_df = pd.read_csv(X_TRAIN_PATH)
    test_df = pd.read_csv(X_TEST_PATH)

    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as file:
        feature_name_map = json.load(file)

    non_feature_columns = ["sample_id", "sample_type", "family_label", "type_label"]

    X_train = train_df.drop(columns=non_feature_columns)
    y_train = train_df["sample_type"]

    X_test = test_df.drop(columns=non_feature_columns)
    y_test = test_df["sample_type"]

    print("Training Random Forest model for feature importance analysis...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    feature_importance_df = pd.DataFrame(
        {
            "feature_id": X_train.columns,
            "feature_name": [
                feature_name_map.get(str(feature_id), f"UNKNOWN:{feature_id}")
                for feature_id in X_train.columns
            ],
            "importance": model.feature_importances_,
        }
    )

    feature_importance_df["feature_category"] = feature_importance_df[
        "feature_name"
    ].apply(get_feature_category)

    feature_importance_df = feature_importance_df.sort_values(
        by="importance", ascending=False
    )

    feature_importance_df.to_csv(CSV_OUTPUT_PATH, index=False)

    top_30 = feature_importance_df.head(30)
    category_summary = (
        feature_importance_df.groupby("feature_category")["importance"]
        .sum()
        .sort_values(ascending=False)
    )

    report = []
    report.append("Random Forest Feature Importance Report")
    report.append("=" * 50)
    report.append("")
    report.append("Task: Binary ransomware detection")
    report.append("Target column: sample_type")
    report.append("Class 0: goodware / benign")
    report.append("Class 1: ransomware")
    report.append("")
    report.append("Model Performance Check")
    report.append("-" * 50)
    report.append(f"Accuracy:  {accuracy:.4f}")
    report.append(f"Precision: {precision:.4f}")
    report.append(f"Recall:    {recall:.4f}")
    report.append(f"F1-score:  {f1:.4f}")
    report.append("")
    report.append("Top 30 Important Behavioural Features")
    report.append("-" * 50)

    for rank, row in enumerate(top_30.itertuples(index=False), start=1):
        report.append(
            f"{rank}. {row.feature_name} "
            f"(feature_id={row.feature_id}, importance={row.importance:.6f})"
        )

    report.append("")
    report.append("Feature Category Importance Summary")
    report.append("-" * 50)
    report.append(str(category_summary))
    report.append("")
    report.append("Output Files")
    report.append("-" * 50)
    report.append(f"Full CSV: {CSV_OUTPUT_PATH}")
    report.append(f"Text report: {TXT_OUTPUT_PATH}")
    report.append("")
    report.append("Safety Note")
    report.append("-" * 50)
    report.append("This analysis uses processed behavioural feature data only.")
    report.append("No live ransomware binaries are downloaded, executed, or handled.")

    report_text = "\n".join(report)

    TXT_OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    print(report_text)
    print("\nFeature importance analysis completed.")


if __name__ == "__main__":
    main()