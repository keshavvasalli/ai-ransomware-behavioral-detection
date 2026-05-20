from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "raw" / "mlran"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "MLRan_X_train_RFE.csv"
X_TEST_PATH = DATA_DIR / "MLRan_X_test_RFE.csv"
LABELS_PATH = DATA_DIR / "MLRan_labels.csv"
METADATA_PATH = DATA_DIR / "mlran_dataset_metadata.csv"
FEATURE_NAMES_PATH = DATA_DIR / "RFE_selected_feature_names_dic.json"

REPORT_PATH = REPORTS_DIR / "mlran_dataset_inspection_report.txt"


def main():
    print("Loading MLRan dataset files...")

    x_train = pd.read_csv(X_TRAIN_PATH)
    x_test = pd.read_csv(X_TEST_PATH)
    labels = pd.read_csv(LABELS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    non_feature_columns = ["sample_id", "sample_type", "family_label", "type_label"]
    feature_columns = [col for col in x_train.columns if col not in non_feature_columns]

    sample_type_distribution = labels["sample_type"].value_counts().sort_index()
    type_label_distribution = labels["type_label"].value_counts().sort_index()
    family_label_count = labels["family_label"].nunique()

    label_meaning = metadata.groupby("sample_type")[["ransomware_family", "ransomware_type"]].first()

    report = []
    report.append("MLRan Dataset Inspection Report")
    report.append("=" * 40)
    report.append("")
    report.append(f"X_train shape: {x_train.shape}")
    report.append(f"X_test shape: {x_test.shape}")
    report.append(f"Labels shape: {labels.shape}")
    report.append(f"Metadata shape: {metadata.shape}")
    report.append("")
    report.append(f"Total samples from train + test: {len(x_train) + len(x_test)}")
    report.append(f"Total label rows: {len(labels)}")
    report.append(f"Total metadata rows: {len(metadata)}")
    report.append("")
    report.append(f"Total columns in X_train: {x_train.shape[1]}")
    report.append(f"Non-feature columns: {non_feature_columns}")
    report.append(f"Detected ML feature columns: {len(feature_columns)}")
    report.append("")
    report.append("Sample type distribution:")
    report.append(str(sample_type_distribution))
    report.append("")
    report.append("Type label distribution:")
    report.append(str(type_label_distribution))
    report.append("")
    report.append(f"Family label count: {family_label_count}")
    report.append("")
    report.append("Verified label meaning from metadata:")
    report.append(str(label_meaning))
    report.append("")
    report.append("Initial ML decision:")
    report.append("Task: Binary classification")
    report.append("Target column: sample_type")
    report.append("Class 0: goodware / benign")
    report.append("Class 1: ransomware")
    report.append("")
    report.append("Safety note:")
    report.append("Only processed behavioural feature files are used.")
    report.append("No live ransomware binaries are downloaded, executed, or handled.")
    report.append("")
    report.append("Feature names JSON top-level type:")
    report.append(str(type(feature_names)))
    report.append("")

    report_text = "\n".join(report)

    print(report_text)

    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()