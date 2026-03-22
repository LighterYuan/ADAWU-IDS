import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(script):
    path = PROJECT_ROOT / "pipelines" / script
    if path.exists():
        print("[RUN]", script)
        subprocess.run([sys.executable, str(path)])
    else:
        print("[SKIP]", script)


def main():
    run("run_baselines.py")
    run("run_paper_trace.py")
    run("run_ablation.py")
    run("run_recovery_eval.py")


if __name__ == "__main__":
    main()
