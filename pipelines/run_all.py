
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_script(rel_path: str):
    script_path = PROJECT_ROOT / rel_path
    print(f"[RUN] {script_path}")
    subprocess.run([sys.executable, str(script_path)], check=True)

if __name__ == "__main__":
    run_script("pipelines/run_baselines.py")
    run_script("pipelines/run_adaptive_baselines.py")
    run_script("analysis/summarize_baseline_comparison.py")
    run_script("visualization/plot_baseline_comparison.py")

