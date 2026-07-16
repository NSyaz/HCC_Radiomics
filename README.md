# HCC Radiomics Classification

This repository provides a reproducible machine-learning workflow for binary hepatocellular carcinoma (HCC) stage-group classification using radiomics features.

The confirmed target column is `Stage`:

- `0` = HCC groups/stages 1-2
- `1` = HCC groups/stages 3-4

The software preserves this mapping and does not reinterpret or remap the target definition. When `--target-column Stage` is used, labels must be exactly `0` and `1`; lossless numeric `0.0` and `1.0` values are canonicalised to integers, while booleans, strings, `{1, 2}`, multiclass labels, missing labels, and single-class datasets are rejected.

## Project Overview

The repository has been recovered and reorganised into a compact scientific Python project. It is designed to let authorised researchers run leakage-aware HCC radiomics experiments locally while keeping real patient-level data out of GitHub.

## Research Objective

The planned modelling comparison focuses on five methods:

- `dummy`: stratified dummy baseline;
- `svm`: SVM without feature selection;
- `svm_kbest`: ANOVA F-test / `SelectKBest` followed by SVM;
- `svm_bpso`: Binary Particle Swarm Optimisation (BPSO) followed by SVM;
- `svm_kbest_bpso`: ANOVA F-test followed by BPSO and SVM.

The scientific objective is to compare these methods on the same train, validation, and test splits without fitting preprocessing, feature selection, BPSO, or model parameters on the final test set.

PR #1 is a repository recovery and infrastructure PR. It is not the final modelling study, and it does not report real-data model performance.

## Dataset Characteristics

The primary analysis dataset is expected to contain:

- approximately 40 patients;
- approximately 662 radiomics predictors;
- one row per patient;
- the binary target column `Stage`;
- no repeated patient rows in the current analysis dataset.

This is a high-dimensional, low-sample-size problem. Real patient-level data are not distributed in this repository.

## Important Scientific Caution

The number of predictors is much larger than the number of patients, so overfitting risk is high. Feature selection and model evaluation must occur strictly within training data or training cross-validation. Single train/test split results should not be overinterpreted. Final scientific reporting should include uncertainty, repeated validation, or nested cross-validation where appropriate.

## Repository Status

- Recovered from historical commit `33aa78fd27bee2ec88f34037beaf104f39fcc64c`.
- Active implementation lives under `src/hcc_radiomics`.
- Historical scripts are preserved under `legacy/` for audit only.
- Synthetic tests pass without private data.
- Real-data experiments have not yet been finalised.
- No real-data model performance is claimed.

## Repository Structure

```text
HCC_Radiomics/
|-- .github/
|   |-- ISSUE_TEMPLATE/
|   |   |-- bug_report.yml
|   |   `-- experiment_request.yml
|   |-- workflows/
|   |   `-- tests.yml
|   `-- pull_request_template.md
|-- configs/
|   `-- default.yaml
|-- data/
|   |-- .gitkeep
|   `-- README.md
|-- docs/
|   |-- experiment_plan.md
|   `-- methodology.md
|-- legacy/
|   |-- README.md
|   |-- Classifiers using BPSO.py
|   |-- SVM (3 Feature Selector).py
|   |-- TPOT (Thesis).py
|   `-- Test Code.py
|-- outputs/
|   `-- .gitkeep
|-- scripts/
|   `-- run_experiment.py
|-- src/
|   `-- hcc_radiomics/
|       |-- __init__.py
|       |-- cli.py
|       |-- data.py
|       |-- evaluation.py
|       |-- feature_selection.py
|       |-- models.py
|       `-- preprocessing.py
|-- tests/
|-- .gitignore
|-- CITATION.cff
|-- README.md
`-- pyproject.toml
```

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Dataset Preparation

Place the authorised local analysis dataset at:

```text
data/hcc_radiomics.xlsx
```

Accepted formats are `.xlsx`, `.xls`, `.csv`, `.tsv`, and tab-delimited `.txt`.

The expected target column is `Stage`:

- `0` = HCC groups/stages 1-2
- `1` = HCC groups/stages 3-4

All approved numeric columns other than `Stage` are treated as candidate radiomics predictors unless excluded with `--metadata-column`. A patient or group column is not required for the current dataset because each row is a different patient, but optional grouped splitting remains available through `--group-column` for future datasets.

## Running Experiments

Dummy baseline:

```bash
python -m hcc_radiomics.cli --data-path data/hcc_radiomics.xlsx --target-column Stage --method dummy --output-dir outputs/dummy
```

Plain SVM:

```bash
python -m hcc_radiomics.cli --data-path data/hcc_radiomics.xlsx --target-column Stage --method svm --output-dir outputs/svm
```

ANOVA F-test plus SVM:

```bash
python -m hcc_radiomics.cli --data-path data/hcc_radiomics.xlsx --target-column Stage --method svm_kbest --k-best 10 --output-dir outputs/svm_kbest
```

BPSO plus SVM:

```bash
python -m hcc_radiomics.cli --data-path data/hcc_radiomics.xlsx --target-column Stage --method svm_bpso --output-dir outputs/svm_bpso
```

ANOVA F-test plus BPSO plus SVM:

```bash
python -m hcc_radiomics.cli --data-path data/hcc_radiomics.xlsx --target-column Stage --method svm_kbest_bpso --k-best 10 --output-dir outputs/svm_kbest_bpso
```

Use the default configuration file:

```bash
python -m hcc_radiomics.cli --config configs/default.yaml
```

## Outputs

Each run creates its output directory and writes:

- `config_used.json`
- `dataset_summary.json`
- `split_class_distributions.json`
- `selected_features.txt`
- `metrics.json`
- `test_predictions.csv`
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `model.joblib`
- `run.log`

## Evaluation Strategy

The current implementation supports leakage-safe train, validation, and final test splitting. Preprocessing, imputation, scaling, ANOVA feature selection, BPSO, and classifier fitting are fitted only on training data or training cross-validation. The final test split is used only for evaluation.

Because the dataset is very small relative to the number of predictors, final real-data conclusions should use repeated stratified validation or nested cross-validation before reporting scientific claims. Nested cross-validation is recommended for the final study design but is not implemented in this repository yet.

## Inference From A Saved Bundle

Each experiment writes `model.joblib`, which stores the full inference bundle: original feature schema and order, excluded metadata columns, fitted preprocessing, fitted ANOVA selector when used, fitted BPSO mask when used, class-label order, final classifier, method name, configuration, and bundle format version.

```python
import pandas as pd
from hcc_radiomics import load_experiment

bundle = load_experiment("outputs/svm_kbest_bpso/model.joblib")
new_data = pd.read_excel("data/hcc_radiomics.xlsx")

predictions = bundle.predict(new_data)
probabilities = bundle.predict_proba(new_data)
```

The bundle accepts a `DataFrame` containing the original radiomics columns, ignores additional unrelated columns, reorders columns to match training, applies the fitted preprocessing and feature selection, and never refits components during inference. Missing required feature columns raise a clear error.

## Testing

Tests use synthetic data only and do not require private research data:

```bash
python -m compileall .
pytest -q
```

## Data Privacy

Real research data are ignored by Git and must remain local. Do not commit Excel workbooks, CSV exports, TSV files, patient identifiers, clinical free text, accessions, dates of birth, credentials, private institutional metadata, or derived row-level patient data.

The repository includes `scripts/check_no_tracked_patient_data.py`, and CI runs it against Git-tracked files to prevent accidental workbook or row-level data commits.

## Next Phase

Before real-data scientific conclusions are reported, the next experiment phase should include:

1. Authorised local dataset validation.
2. Class-distribution verification.
3. Repeated stratified nested cross-validation.
4. Inner-loop feature selection and hyperparameter tuning.
5. Feature-selection stability analysis.
6. Confidence intervals.
7. Comparison against the stratified dummy baseline.
8. A locked final reporting protocol.

## Legacy Code

The original scripts are retained in `legacy/` so reviewers can audit the recovered history. They contain hard-coded local paths and known leakage-prone modelling patterns, so they must not be used as the active pipeline.

## Citation

Citation metadata are provided in `CITATION.cff`. No manuscript details, DOI, journal name, institution, or author affiliation are available in the repository at this time.

## Licence

No licence has yet been selected. All rights remain with the repository owner unless an explicit licence is later added.
