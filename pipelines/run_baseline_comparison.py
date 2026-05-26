from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_py(script: Path, *args):
    cmd = [sys.executable, str(script), *map(str, args)]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed adaptive baselines and build summary")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    seeds = cfg.get("seeds", [cfg.get("seed", 42)])

    adaptive_runner = PROJECT_ROOT / "pipelines" / "run_adaptive_baselines.py"
    summary_builder = PROJECT_ROOT / "analysis" / "build_baseline_comparison_summary.py"

    for seed in seeds:
        run_py(adaptive_runner, "--config", args.config, "--seed", seed)

    run_py(summary_builder)
    print("[OK] full baseline comparison pipeline complete")


if __name__ == "__main__":
    main()
