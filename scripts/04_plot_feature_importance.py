from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = BASE_DIR / "reports"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "random_forest_feature_importance.csv"

TOP_FEATURES_CHART = REPORTS_DIR / "top_20_random_forest_feature_importance.png"
CATEGORY_CHART = REPORTS_DIR / "feature_category_importance.png"


def shorten_label(text: str, max_length: int = 55) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def main():
    print("Loading feature importance CSV...")

    df = pd.read_csv(FEATURE_IMPORTANCE_CSV)

    print("Creating top 20 feature importance chart...")

    top_20 = df.head(20).copy()
    top_20["short_feature_name"] = top_20["feature_name"].apply(shorten_label)

    plt.figure(figsize=(12, 8))
    plt.barh(top_20["short_feature_name"][::-1], top_20["importance"][::-1])
    plt.xlabel("Feature Importance")
    plt.ylabel("Behavioural Feature")
    plt.title("Top 20 Random Forest Behavioural Feature Importances")
    plt.tight_layout()
    plt.savefig(TOP_FEATURES_CHART, dpi=300)
    plt.close()

    print(f"Top 20 feature chart saved to: {TOP_FEATURES_CHART}")

    print("Creating feature category importance chart...")

    category_summary = (
        df.groupby("feature_category")["importance"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    plt.bar(category_summary.index, category_summary.values)
    plt.xlabel("Feature Category")
    plt.ylabel("Total Importance")
    plt.title("Random Forest Feature Importance by Behavioural Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(CATEGORY_CHART, dpi=300)
    plt.close()

    print(f"Feature category chart saved to: {CATEGORY_CHART}")

    print("Feature importance visualisation completed.")


if __name__ == "__main__":
    main()