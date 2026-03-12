"""
实验脚本 - 运行基线与动态集成学习框架并记录结果
"""

import os
import json
import numpy as np

from config import Config
from training.trainer import ModelTrainer
from training.ensemble_trainer import EnsembleTrainer
from data.data_processor import CICIDSDataProcessor


def load_preprocessed(file_base):
    processed_dir = Config.PROCESSED_DATA_DIR
    X_path = os.path.join(processed_dir, f'{file_base}_X.npy')
    y_path = os.path.join(processed_dir, f'{file_base}_y.npy')
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"预处理数据不存在: {file_base}")

    X = np.load(X_path)
    y = np.load(y_path)
    if X.shape[2] != 78:
        X = X[:, :, :78]
    return X, y


def run_baseline(train_file, eval_files):
    trainer = ModelTrainer()
    X_train, y_train = load_preprocessed(train_file)

    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    trainer.train_basic_model(
        (X_train_split, y_train_split),
        (X_val_split, y_val_split),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )

    results = {}
    for file in eval_files:
        X_test, y_test = load_preprocessed(file)
        eval_res = trainer.evaluate_model((X_test, y_test), save_results=False)
        results[file] = {
            'accuracy': eval_res['accuracy'],
            'f1': eval_res['classification_report'].get('1', {}).get('f1-score', 0.0)
        }
    return results


def run_dynamic_ensemble(train_file, eval_files, ensemble_type='dynamic', n_models=3):
    trainer = EnsembleTrainer(use_ensemble=True, ensemble_type=ensemble_type)
    X_train, y_train = load_preprocessed(train_file)

    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    trainer.create_base_model(input_shape=(Config.SEQUENCE_LENGTH, 78))
    trainer.setup_ensemble(n_models=n_models, drift_detector_type='ensemble')
    trainer.train_initial_models(
        (X_train_split, y_train_split),
        (X_val_split, y_val_split),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE
    )

    processor = CICIDSDataProcessor()
    results = {}
    for file in eval_files:
        X_test, y_test = load_preprocessed(file)
        eval_res = trainer.adaptive_evaluation((X_test, y_test), batch_size=1000)
        avg_accuracy = float(np.mean(eval_res['batch_accuracies'])) if eval_res['batch_accuracies'] else 0.0
        drift_ratio = (
            sum(eval_res['drift_detections']) / len(eval_res['drift_detections'])
            if eval_res['drift_detections'] else 0.0
        )
        results[file] = {
            'accuracy': avg_accuracy,
            'drift_ratio': drift_ratio,
            'final_weights': eval_res['ensemble_weights'][-1] if eval_res['ensemble_weights'] else []
        }
    return results


def main():
    Config.create_directories()
    processor = CICIDSDataProcessor()
    ordered_files = processor.get_ordered_csv_files(Config.CIC_IDS_DIR)
    if len(ordered_files) < 4:
        raise ValueError("预处理数据不足，无法运行实验")

    train_file = os.path.splitext(ordered_files[0])[0]
    eval_files = [os.path.splitext(f)[0] for f in ordered_files[1:4]]

    print("运行基线模型实验...")
    baseline_results = run_baseline(train_file, eval_files)

    print("运行动态集成学习实验...")
    ensemble_results = run_dynamic_ensemble(train_file, eval_files)

    summary = {
        'train_file': train_file,
        'eval_files': eval_files,
        'baseline': baseline_results,
        'dynamic_ensemble': ensemble_results
    }

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(Config.RESULTS_DIR, 'experiment_summary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"实验结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

