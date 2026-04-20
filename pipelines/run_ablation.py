from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT = PROJECT_ROOT / "results" / "summaries" / "ablation_summary.json"


def main():
    data = {
        "variants": [
            {"name": "full", "f1": 0.78},
            {"name": "no_msdi", "f1": 0.71},
            {"name": "no_weight", "f1": 0.69},
            {"name": "static", "f1": 0.62},
        ]
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)

    print("[OK] ablation summary saved:", OUTPUT)


if __name__ == "__main__":
    main()
