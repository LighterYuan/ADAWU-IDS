# ADAWU-IDS Reproducibility Guide

This repository provides the code used for the ADAWU-IDS chronological concept-drift-aware intrusion detection experiments. The guide below is intended to make the main experiments reproducible and to clarify the exact training, validation, and testing protocol.

## 1. Repository structure

```text
ADAWU-IDS/
├── configs/
│   ├── paper_main.yaml
│   ├── baselines.yaml
│   ├── ablations.yaml
│   └── recovery.yaml
├── drift/
│   ├── concept_drift_detector.py
│   ├── dynamic_ensemble.py
│   └── adaptive_learning.py
├── models/
│   └── lstm_model.py
├── baselines/
│   ├── dwm.py
│   ├── online_bagging.py
│   └── leveraging_bagging.py
├── pipelines/
│   ├── run_calibrated_cicids2017_experiment.py
│   ├── run_fair_cicids2017_experiment.py
│   ├── run_ablation.py
│   ├── run_adaptive_baselines.py
│   ├── run_efficiency_eval.py
│   └── run_unsw_ablation.py
├── analysis/
│   ├── aggregate_runs.py
│   ├── build_ablation_summary.py
│   ├── build_drift_summary.py
│   ├── build_efficiency_summary.py
│   └── generate_ablation_paper_figures.py
└── paper_outputs/
```

## 2. Environment

The code was developed for Python 3. Recommended core packages are:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn pyyaml tensorflow
```

A typical environment is:

```text
Python >= 3.9
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
PyYAML
TensorFlow / Keras
```

For deterministic reruns, set the random seeds reported in the manuscript and in the command examples below.

## 3. Dataset preparation

### 3.1 CICIDS2017

The main chronological experiment uses CICIDS2017. The expected segments are:

```text
Training stream:
- Tuesday
- Wednesday

Validation stream:
- Thursday WebAttacks

Final test stream:
- Thursday Infiltration
- Friday Morning
- Friday PortScan
```

Place the processed CICIDS2017 segment files under:

```text
datasets/processed/
```

The scripts can read segment files in several formats, including `.npz`, `.npy` X/y pairs, `.csv`, or `.parquet`, as long as the file names contain recognizable segment names. Recognized aliases include:

```text
Tuesday
Tuesday-WorkingHours

Wednesday
Wednesday-workingHours
Wednesday-WorkingHours

Thursday WebAttacks
Thursday_WebAttacks
WebAttacks
Thursday-WorkingHours-Morning-WebAttacks
Thursday-WorkingHours-Afternoon-WebAttacks

Thursday Infiltration
Thursday_Infiltration
Infiltration
Infilteration
Thursday-WorkingHours-Afternoon-Infilteration
Thursday-WorkingHours-Afternoon-Infiltration

Friday Morning
Friday_Morning
Friday-WorkingHours-Morning

Friday PortScan
Friday_PortScan
PortScan
Friday-WorkingHours-Afternoon-PortScan
```

For `.npz` files, the script searches for feature keys such as `X`, `features`, or `data`, and label keys such as `y`, `label`, `labels`, `target`, or `targets`. For `.csv` files, labels should be stored in one of the following columns:

```text
Label, label, labels, target, Target, class, Class, y
```

The preprocessing inside the experiment fits imputers and scalers only on the initial training stream. The validation and test streams are transformed using the training-fitted preprocessing objects to avoid information leakage.

### 3.2 UNSW-NB15

UNSW-NB15 is used as a secondary cross-dataset/generalization analysis. It is not used as the primary validation-calibrated chronological experiment because it does not provide the same multi-day chronological validation/test structure as CICIDS2017.

Expected raw files for the standalone UNSW-NB15 experiment are:

```text
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

Example command:

```bash
python unsw_nb15_full_experiment.py \
  --train path/to/UNSW_NB15_training-set.csv \
  --test path/to/UNSW_NB15_testing-set.csv \
  --outdir results/unsw_nb15 \
  --task binary \
  --chunk-size 5000 \
  --seeds 42 52 62
```

## 4. Main chronological CICIDS2017 experiment

The main calibrated CICIDS2017 experiment is implemented in:

```text
pipelines/run_calibrated_cicids2017_experiment.py
```

It follows this protocol:

```text
Train:      Tuesday + Wednesday
Validation: Thursday WebAttacks
Test:       Thursday Infiltration + Friday Morning + Friday PortScan
```

The validation stream is used only to select the ADAWU response configuration. The final test stream is never used to select coefficients, thresholds, or response actions.

Example command:

```bash
python pipelines/run_calibrated_cicids2017_experiment.py \
  --data-dir datasets/processed \
  --output-dir results/calibrated_cicids2017 \
  --seeds 42 52 62 \
  --calibration-seeds 42 \
  --chunk-size 10000 \
  --candidate-grid small \
  --selection-metric attack_sensitive \
  --epochs 3 \
  --online-epochs 1 \
  --batch-size 512 \
  --include-ablation
```

Expected outputs include:

```text
results/calibrated_cicids2017/
├── calibration/
│   ├── selected_candidate.json
│   └── candidate_validation_summary.csv
├── per_seed/
│   ├── <method>_seed<seed>_summary.json
│   └── <method>_seed<seed>_chunks.csv
├── tables/
│   ├── overall_mean_std_ci95.csv
│   └── paired_statistical_tests.csv
└── protocol_manifest.json
```

## 5. Fair chronological comparison with online ensemble baselines

The fair CICIDS2017 comparison with Static LSTM, ADAWU-IDS, Dynamic Weighted Majority, Online Bagging, and Leveraging Bagging is implemented in:

```text
pipelines/run_fair_cicids2017_experiment.py
```

Example command:

```bash
python pipelines/run_fair_cicids2017_experiment.py \
  --data-dir datasets/processed \
  --output-dir results/fair_cicids2017 \
  --seeds 42 52 62 \
  --chunk-size 5000 \
  --epochs 5 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --adaptation-rate 1e-4 \
  --alpha 0.60 \
  --beta 0.25 \
  --gamma 0.15 \
  --lambda-decay 0.10 \
  --mild-threshold 0.30 \
  --moderate-threshold 0.50 \
  --severe-threshold 0.70 \
  --include-ablation
```

This script uses the same prediction-before-update protocol for all adaptive methods. For each incoming chunk, the method first predicts the chunk labels using the current model state. Only after prediction are the current labels used for evaluation and online updating.

## 6. ADAWU parameter settings

The main paper configuration is stored in:

```text
configs/paper_main.yaml
```

The key ADAWU parameters in this configuration are:

```yaml
drift:
  msdi_threshold: 0.30
  mild_threshold: 0.30
  moderate_threshold: 0.50
  severe_threshold: 0.70

ensemble:
  n_models: 3
  alpha: 0.60
  beta: 0.25
  gamma: 0.15
  lambda_decay: 0.10
  min_weight: 0.05
```

Therefore, the values used for the manuscript parameter-calibration table are:

```text
w_min = 0.05
MSDI thresholds = 0.30 / 0.50 / 0.70
```

The calibrated experiment also includes candidate response configurations in `pipelines/run_calibrated_cicids2017_experiment.py`. The response policy is selected only on the Thursday WebAttacks validation stream before final testing.

## 7. Validation-only calibration

The candidate response policies are evaluated only on the validation stream. The default selection metric is:

```text
attack_sensitive
```

This metric combines overall and attack-sensitive terms, including Weighted F1, Macro F1, Attack F1, Attack Recall, and false-negative rate. It favors configurations that improve attack detection while penalizing missed attacks.

The final test stream is not used to:

```text
- choose alpha, beta, gamma, or lambda_decay;
- choose MSDI thresholds;
- choose minimum weight constraints;
- enable or disable hierarchical retraining;
- select the validation scoring rule.
```

After the validation-selected configuration is fixed, it is applied unchanged to the final chronological test stream.

## 8. Ablation study

Component-level ablation is implemented in:

```text
pipelines/run_ablation.py
```

Example command:

```bash
python pipelines/run_ablation.py \
  --data-dir datasets/processed \
  --dataset CICIDS2017 \
  --seed 42 \
  --chunk-size 5000 \
  --reference-chunks 2 \
  --initial-train-chunks 3 \
  --variant all \
  --output-dir results/traces/ablations
```

The ablation variants include:

```text
full_adawu_ids
w_o_msdi
w_o_dynamic_weighting
w_o_hierarchical_response
static_lstm_or_static_sgd
```

The ablation results are intended to show both useful and harmful component interactions. In the revised manuscript, hierarchical retraining is treated as an optional validation-selected response action rather than as an always-beneficial mandatory component.

## 9. Drift-detection diagnostics

Drift-related analysis is implemented through the MSDI and drift-monitoring code in:

```text
drift/concept_drift_detector.py
drift/dynamic_ensemble.py
pipelines/run_calibrated_cicids2017_experiment.py
```

In the revised manuscript, MSDI is interpreted as a drift-severity and diagnostic signal rather than as a complete standalone binary drift detector. Drift diagnostics should therefore be interpreted as supporting evidence for monitoring and adaptive control, not as proof of a deployment-ready autonomous drift alarm.

## 10. Computational-cost analysis

Efficiency analysis can be generated with:

```bash
python pipelines/run_efficiency_eval.py \
  --input results/traces/paper_trace_6_5.json \
  --output results/traces/paper_trace_6_6.json
```

The generated outputs can be summarized using the scripts in `analysis/`.

## 11. Reproducing tables and figures

The following scripts are used to aggregate outputs and generate paper-level summaries:

```text
analysis/aggregate_runs.py
analysis/build_ablation_summary.py
analysis/build_baseline_comparison_summary.py
analysis/build_drift_summary.py
analysis/build_efficiency_summary.py
analysis/build_post_drift_analysis_paper.py
analysis/build_recovery_summary.py
analysis/build_temporal_summary.py
analysis/build_weight_summary.py
analysis/generate_ablation_paper_figures.py
```

Typical workflow:

```bash
# 1. Run the main calibrated experiment.
python pipelines/run_calibrated_cicids2017_experiment.py \
  --data-dir datasets/processed \
  --output-dir results/calibrated_cicids2017 \
  --seeds 42 52 62 \
  --calibration-seeds 42 \
  --chunk-size 10000 \
  --candidate-grid small \
  --selection-metric attack_sensitive \
  --epochs 3 \
  --online-epochs 1 \
  --batch-size 512 \
  --include-ablation

# 2. Run the fair comparison with online ensemble baselines.
python pipelines/run_fair_cicids2017_experiment.py \
  --data-dir datasets/processed \
  --output-dir results/fair_cicids2017 \
  --seeds 42 52 62 \
  --chunk-size 5000 \
  --include-ablation

# 3. Run ablation analysis, if needed.
python pipelines/run_ablation.py \
  --data-dir datasets/processed \
  --dataset CICIDS2017 \
  --variant all \
  --seed 42 \
  --output-dir results/traces/ablations
```

## 12. Notes on interpretation

The code and experiments support the revised positioning of ADAWU-IDS:

```text
- ADAWU-IDS is not claimed to be a universally superior IDS classifier.
- MSDI is treated as a drift-severity signal, not a standalone binary detector.
- Hierarchical retraining is treated as an optional validation-selected response action.
- Classical online ensemble baselines may achieve stronger raw predictive performance.
- The main contribution is a transparent chronological drift-response pipeline with explicit calibration, ablation, statistical testing, and failure-case reporting.
```

## 13. Troubleshooting

### Segment file not found

Check that file names contain one of the expected segment aliases listed in Section 3.

### Label column not found

For CSV input, ensure that the label column is named one of:

```text
Label, label, labels, target, Target, class, Class, y
```

### TensorFlow unavailable

If TensorFlow is not installed, install it with:

```bash
pip install tensorflow
```

### Memory issues

Use a smaller chunk size or reduce training samples for debugging:

```bash
--chunk-size 2000
--max-train-samples 50000
```

## 14. Citation note

When using this repository to reproduce the manuscript results, please report the chronological split, random seeds, chunk size, selected configuration, and whether hierarchical retraining is enabled or disabled.
