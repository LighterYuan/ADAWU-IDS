from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_STATIC = PROJECT_ROOT / "results" / "cases" / "static_predictions.npz"
INPUT_ADAWU = PROJECT_ROOT / "results" / "cases" / "adawu_predictions.npz"

OUTPUT_SUMMARY = PROJECT_ROOT / "results" / "summaries" / "temporal_comparison_summary.json"


SEGMENTS = [
    "Tuesday-WorkingHours",
    "Wednesday-WorkingHours",
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Thursday-WorkingHours-Afternoon-Infiltration",
    "Friday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
]


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def main():
    static = np.load(INPUT_STATIC, allow_pickle=True)
    adawu = np.load(INPUT_ADAWU, allow_pickle=True)

    segments = static["segments"]

    results = []

    for seg in SEGMENTS:
        mask = segments == seg

        y_true = static["y_true"][mask]
        y_pred_static = static["y_pred"][mask]
        y_pred_adawu = adawu["y_pred"][mask]

        m_static = compute_metrics(y_true, y_pred_static)
        m_adawu = compute_metrics(y_true, y_pred_adawu)

        results.append({
            "segment": seg,
            "static_accuracy": m_static["accuracy"],
            "adawu_accuracy": m_adawu["accuracy"],
            "static_weighted_f1": m_static["weighted_f1"],
            "adawu_weighted_f1": m_adawu["weighted_f1"]
        })

    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump({"segments": results}, f, indent=2)

    print("[OK] summary saved:", OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
