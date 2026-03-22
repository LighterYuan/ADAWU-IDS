# ADAWU-IDS

ADAWU-IDS is a concept drift-aware network intrusion detection project built around an LSTM-based detector and an adaptive ensemble update mechanism for non-stationary traffic analysis.

This repository is intended to be used as a project manual. It explains the project purpose, major modules, data flow, and execution process for running the system and generating outputs.

## 1. Project Function

The project provides an end-to-end workflow for network intrusion detection under changing data distributions. Its main functions include:

- loading and organizing processed traffic data,
- training LSTM-based base learners,
- running baseline and adaptive detection pipelines,
- monitoring drift-related signals during temporal evaluation,
- updating ensemble behavior through adaptive weighting,
- generating structured results, summaries, figures, and tables.

In practical terms, the repository supports both model development and experiment execution, while also keeping result artifacts organized for later inspection.

## 2. Core Implementation Idea

The implementation follows a modular pipeline.

1. **Data preparation**
   - Processed feature arrays are loaded from the dataset directory.
   - Inputs are organized into temporal sequences for the LSTM model.

2. **Base model training**
   - LSTM base learners are trained on prepared sequence data.
   - Training utilities handle batching, optimization, and validation.

3. **Temporal evaluation**
   - The trained models are evaluated on later data segments in chronological order.
   - Prediction behavior is recorded over time.

4. **Adaptive update**
   - Drift-related statistics and model behavior are monitored during evaluation.
   - Ensemble weights are updated dynamically so the system can respond to distribution changes.

5. **Result generation**
   - Predictions, traces, summaries, figures, and tables are written into the `results/` directory.
   - Visualization and analysis scripts convert raw outputs into structured artifacts.

## 3. Repository Structure

```text
ADAWU-IDS/
├── analysis/                # Post-processing and summary scripts
├── configs/                 # Configuration files
├── drift/                   # Drift-aware logic and adaptive update modules
├── legacy/                  # Older code kept for reference
├── models/                  # Model definitions
├── pipelines/               # Main execution entry points
├── results/                 # Generated outputs
│   ├── cases/               # Prediction-level outputs
│   ├── figures/             # Generated figures
│   ├── summaries/           # Summary files
│   ├── tables/              # Generated tables
│   └── traces/              # Intermediate trace records
├── training/                # Training utilities
└── visualization/           # Figure and table rendering scripts
```

## 4. Main Modules

### `models/`
Contains the model definitions used by the project, primarily the LSTM-based classifier.

### `training/`
Contains training helpers, trainers, and related utilities for fitting models.

### `drift/`
Contains the adaptive logic used to react to changing data behavior, including weight adjustment and drift-related processing.

### `pipelines/`
Contains the main runnable scripts for executing the project workflow.

### `analysis/`
Contains scripts that transform raw outputs into structured summaries.

### `visualization/`
Contains scripts for converting summaries or traces into figures and tables.

### `configs/`
Contains configuration files controlling data paths, model settings, runtime behavior, and adaptive parameters.

### `results/`
Stores all generated outputs produced by the pipelines and analysis scripts.

## 5. Environment Requirements

Recommended environment:

- Python 3.10 or newer
- TensorFlow 2.x
- NumPy
- pandas
- scikit-learn
- SciPy
- matplotlib

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn matplotlib tensorflow
```

On Windows:

```bash
.venv\Scripts\activate
```

## 6. Data Requirements

The project expects processed data files to be placed under:

```text
datasets/processed/
```

Typical file pattern:

```text
<segment_name>_X.npy
<segment_name>_y.npy
```

General expectations:

- `X` contains feature data
- `y` contains labels
- input data should be numerically encoded and ready for model consumption
- feature dimensions must remain consistent across training and evaluation stages

If 2D feature arrays are used, some scripts may expand them automatically into sequence-compatible shapes.

## 7. Configuration

Main runtime behavior is controlled by configuration files under:

```text
configs/
```

These files typically define:

- dataset paths,
- model parameters,
- optimizer settings,
- training settings,
- adaptive update parameters,
- runtime options,
- output locations.

Before running the project, review the configuration files and confirm that local paths and parameter settings match your environment.

## 8. Execution Flow

A standard usage flow is:

### Step 1: Prepare processed data
Place the required processed `.npy` data files into `datasets/processed/`.

### Step 2: Review configuration
Open the relevant configuration file under `configs/` and verify:
- data paths,
- model settings,
- runtime options,
- output behavior.

### Step 3: Run the baseline or main pipeline
Use a script in `pipelines/` to launch the desired workflow.

Examples:

```bash
python pipelines/run_baselines.py
python pipelines/run_paper_trace.py
python pipelines/run_all.py
```

The exact scripts available may vary by project version, but the main entry points are located in the `pipelines/` directory.

### Step 4: Inspect generated outputs
After execution, results are written into:

```text
results/
```

You can then inspect:
- case outputs,
- traces,
- summaries,
- figures,
- tables.

### Step 5: Run analysis or visualization scripts if needed
To convert raw outputs into more interpretable artifacts, run scripts under:

```text
analysis/
visualization/
```

## 9. Output Structure

The project organizes outputs into several layers:

- **cases**: direct prediction outputs or saved result bundles
- **traces**: intermediate runtime records
- **summaries**: condensed structured results
- **figures**: visual outputs
- **tables**: tabular outputs

This design helps separate raw execution results from post-processed artifacts.

## 10. Typical Workflow Example

A typical implementation workflow is:

1. prepare processed dataset files,
2. confirm configuration settings,
3. run a training or evaluation pipeline,
4. generate traces and saved predictions,
5. build summaries from raw results,
6. render figures or tables if needed,
7. inspect outputs under `results/`.

This makes the project easier to run, debug, and extend.

## 11. Troubleshooting

### Missing data files
If a script reports missing `.npy` files:
- check that the files exist under `datasets/processed/`,
- confirm the filenames match what the scripts expect,
- make sure you are running commands from the repository root.

### Dependency issues
If Python reports missing packages, install the required libraries listed above.

### TensorFlow execution problems
If GPU support is unavailable, the project may still run on CPU, although execution can be slower.

### Shape mismatch
If model inputs have incompatible shapes:
- verify feature dimensions are consistent,
- confirm labels are valid,
- check whether sequence formatting is required by the current pipeline.

## 12. Extension Guidance

To extend the project safely:

- add new pipeline scripts under `pipelines/`,
- add new model definitions under `models/`,
- add new analysis logic under `analysis/`,
- add new figure or table builders under `visualization/`,
- keep configuration changes centralized in `configs/`.

This structure helps preserve readability and maintainability.
