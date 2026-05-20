from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = BASE_DIR / "reports"
METRICS_CSV = REPORTS_DIR / "model_comparison_metrics.csv"
OUTPUT_CHART = REPORTS_DIR / "model_comparison_metrics.png"


def main():
    print("Loading model comparison metrics...")

    df = pd.read_csv(METRICS_CSV)

    metric_columns = ["accuracy", "precision", "recall", "f1_score"]

    chart_df = df.set_index("model")[metric_columns]

    print("Creating model comparison chart...")

    ax = chart_df.plot(kind="bar", figsize=(10, 6))

    ax.set_title("Model Comparison for Binary Ransomware Detection")
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=300)
    plt.close()

    print(f"Model comparison chart saved to: {OUTPUT_CHART}")
    print("Model comparison visualisation completed.")


if __name__ == "__main__":
    main()