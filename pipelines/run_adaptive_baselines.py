from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import TEST_SEGMENTS, TRAIN_SEGMENTS, load_xy
from baselines.static_lstm import StaticLSTMConfig, StaticLSTMOnlineEval
from baselines.dwm_lstm import DWMConfig, DWMLSTM
from baselines.online_bagging_lstm import OnlineBaggingConfig, OnlineBaggingLSTM
from baselines.drift_triggered_ensemble import DriftTriggeredConfig, DriftTriggeredEnsemble
from adawu.adawu_ids import ADAWUConfig, ADAWUIDS


BASELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "adaptive_baselines.yaml"
ADAWU_CONFIG_PATH = PROJECT_ROOT / "configs" / "adawu.yaml"
CASES_DIR = PROJECT_ROOT / "results" / "cases"
TRACES_DIR = PROJECT_ROOT / "results" / "traces"


def save_npz(name: str, y_true: np.ndarray, y_pred: np.ndarray, segments: np.ndarray) -> None:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CASES_DIR / f"{name}_predictions.npz"
    np.savez(out_path, y_true=y_true, y_pred=y_pred, segments=segments)
    print(f"[OK] saved {out_path}")


def save_trace(name: str, trace) -> None:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRACES_DIR / f"{name}_trace.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(f"[OK] saved {out_path}")


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_static(cfg):
    model = StaticLSTMOnlineEval(
        StaticLSTMConfig(
            init_epochs=cfg["common"]["init_epochs"],
            batch_size=cfg["common"]["batch_size"],
            random_state=cfg["common"]["random_state"],
            chunk_size=cfg["common"]["chunk_size"],
        )
    )
    model.fit_initial(TRAIN_SEGMENTS)
    return model.evaluate_stream(TEST_SEGMENTS, load_xy)


def run_dwm(cfg):
    model = DWMLSTM(
        DWMConfig(
            beta=cfg["dwm"]["beta"],
            theta=cfg["dwm"]["theta"],
            max_learners=cfg["dwm"]["max_learners"],
            update_epochs=cfg["common"]["update_epochs"],
            init_epochs=cfg["common"]["init_epochs"],
            batch_size=cfg["common"]["batch_size"],
            random_state=cfg["common"]["random_state"],
            chunk_size=cfg["common"]["chunk_size"],
        )
    )
    model.fit_initial(TRAIN_SEGMENTS)
    return model.evaluate_stream(TEST_SEGMENTS, load_xy)


def run_online_bagging(cfg):
    model = OnlineBaggingLSTM(
        OnlineBaggingConfig(
            n_estimators=cfg["online_bagging"]["n_estimators"],
            lam=cfg["online_bagging"]["lam"],
            init_epochs=cfg["common"]["init_epochs"],
            update_epochs=cfg["common"]["update_epochs"],
            batch_size=cfg["common"]["batch_size"],
            random_state=cfg["common"]["random_state"],
            chunk_size=cfg["common"]["chunk_size"],
        )
    )
    model.fit_initial(TRAIN_SEGMENTS)
    return model.evaluate_stream(TEST_SEGMENTS, load_xy)


def run_drift_triggered(cfg):
    model = DriftTriggeredEnsemble(
        DriftTriggeredConfig(
            n_estimators=cfg["drift_triggered"]["n_estimators"],
            init_epochs=cfg["common"]["init_epochs"],
            update_epochs=cfg["common"]["update_epochs"],
            batch_size=cfg["common"]["batch_size"],
            drift_error_threshold=cfg["drift_triggered"]["drift_error_threshold"],
            performance_drop_threshold=cfg["drift_triggered"]["performance_drop_threshold"],
            memory_chunks=cfg["drift_triggered"]["memory_chunks"],
            random_state=cfg["common"]["random_state"],
            chunk_size=cfg["common"]["chunk_size"],
        )
    )
    model.fit_initial(TRAIN_SEGMENTS)
    return model.evaluate_stream(TEST_SEGMENTS, load_xy)


def run_adawu(cfg):
    model = ADAWUIDS(
        ADAWUConfig(
            n_estimators=cfg["common"]["n_estimators"],
            init_epochs=cfg["common"]["init_epochs"],
            update_epochs_moderate=cfg["response"]["update_epochs_moderate"],
            update_epochs_severe=cfg["response"]["update_epochs_severe"],
            batch_size=cfg["common"]["batch_size"],
            random_state=cfg["common"]["random_state"],
            chunk_size=cfg["common"]["chunk_size"],
            alpha=cfg["adawu"]["alpha"],
            beta=cfg["adawu"]["beta"],
            gamma=cfg["adawu"]["gamma"],
            lam=cfg["adawu"]["lam"],
            min_weight=cfg["adawu"]["min_weight"],
            eta=cfg["msdi"]["eta"],
            mild_threshold=cfg["response"]["mild_threshold"],
            moderate_threshold=cfg["response"]["moderate_threshold"],
            severe_threshold=cfg["response"]["severe_threshold"],
            reference_chunks=cfg["common"]["reference_chunks"],
        )
    )
    model.fit_initial(TRAIN_SEGMENTS)
    return model.evaluate_stream(TEST_SEGMENTS, load_xy)def run_adawu(cfg):
    model = ADAWUIDS(
        ADAWUConfig(
            n_estimators=cfg["common"]["n_estimators"],
            init_epochs=cfg["common"]["init_epochs"],
            update_epochs_moderate=cfg["response"]["update_epochs_moderate"],
            update_epochs_severe=cfg["response"]["update_epochs_severe"],
            batch_size=cfg["common"]["batch_size"],
            random_state=cfg["common"]["random_state"],
            chunk_size=cfg["common"]["chunk_size"],
            alpha=cfg["adawu"]["alpha"],
            beta=cfg["adawu"]["beta"],
            gamma=cfg["adawu"]["gamma"],
            lam=cfg["adawu"]["lam"],
            min_weight=cfg["adawu"]["min_weight"],
            eta=cfg["msdi"]["eta"],
            mild_threshold=cfg["response"]["mild_threshold"],
            moderate_threshold=cfg["response"]["moderate_threshold"],
            severe_threshold=cfg["response"]["severe_threshold"],
            reference_chunks=cfg["common"]["reference_chunks"],
            severe_boost_factor=cfg["response"]["severe_boost_factor"],
            moderate_boost_factor=cfg["response"]["moderate_boost_factor"],
        )
    )
    model.fit_initial(TRAIN_SEGMENTS)
    return model.evaluate_stream(TEST_SEGMENTS, load_xy)



def main():
    baseline_cfg = load_yaml(BASELINE_CONFIG_PATH)
    adawu_cfg = load_yaml(ADAWU_CONFIG_PATH)

    runners = {
        "static_chunkwise": lambda: run_static(baseline_cfg),
        "dwm": lambda: run_dwm(baseline_cfg),
        "online_bagging": lambda: run_online_bagging(baseline_cfg),
        "drift_triggered": lambda: run_drift_triggered(baseline_cfg),
        "adawu": lambda: run_adawu(adawu_cfg),
    }

    for name, fn in runners.items():
        print(f"\n=== Running {name} ===")
        y_true, y_pred, segments, trace = fn()
        save_npz(name, y_true, y_pred, segments)
        save_trace(name, trace)


if __name__ == "__main__":
    main()
