# HCC Radiomics Classification

This repository contains a rebuilt, leakage-aware machine-learning workflow for hepatocellular carcinoma (HCC) radiomics classification. The historical project intent was to classify HCC stage from radiomics features using SVM models with ANOVA F-test / `SelectKBest` and Binary Particle Swarm Optimisation (BPSO) feature selection.

## Project Status

The repository was recovered from historical commit `33aa78fd27bee2ec88f34037beaf104f39fcc64c` after later commits removed the uploaded project files. The active implementation has been restructured into a Python package under `src/hcc_radiomics`. Historical scripts are preserved in `legacy/` for audit only and are not the supported workflow.

Real-data validation has not been performed in this rebuild because the recovered Excel workbooks contain row-level research data and were deliberately excluded from Git.

## Repository Structure

```text
.
├── configs/default.yaml
├── data/README.md
├── legacy/
├── outputs/.gitkeep
├── scripts/run_experiment.py
├── src/hcc_radiomics/
│   ├── cli.py
│   ├── data.py
│   ├── evaluation.py
│   ├── feature_selection.py
│   ├── models.py
│   └── preprocessing.py
└── tests/
```

## Research Objective

The implemented workflow supports supervised classification of HCC stage labels from numeric radiomics features. It is intended to help researchers run reproducible experiments while preventing common sources of optimistic bias, especially feature-selection and model-fitting leakage from the final test set.

## Machine-Learning Workflow

The pipeline:

1. loads a local tabular dataset;
2. validates the target column and numeric feature columns;
3. excludes target, identifier, and metadata columns from predictors;
4. creates reproducible train, validation, and final test splits;
5. fits imputation, zero-variance filtering, and scaling on training data only;
6. optionally applies ANOVA `SelectKBest`;
7. optionally applies BPSO feature selection using training cross-validation only;
8. trains an SVM classifier on the selected training features;
9. evaluates validation and final test splits without refitting on test data;
10. saves metrics, figures, selected features, predictions, configuration, and model artifacts.

## Dataset Requirements

Provide an authorised local `.xlsx`, `.csv`, or `.tsv` dataset. The default target column is `Stage`, but this can be changed with `--target-column`.

Required columns:

- a target column such as `Stage`;
- numeric radiomics feature columns.

Optional columns:

- patient or case identifiers;
- clinical or acquisition metadata;
- a grouping column for repeated patient records.

Pass identifiers and metadata with `--metadata-column`. If multiple rows can belong to the same patient, pass `--group-column` so grouped splitting prevents patient overlap between train, validation, and test sets.

## Privacy And Governance

Do not commit patient identifiers, protected health information, accession numbers, dates of birth, clinical free text, credentials, private institutional metadata, or row-level research data. The historical data workbooks were inspected and excluded from this rebuild. Authorised researchers should place approved local data under `data/` after confirming ethics, governance, and data-sharing requirements.

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Example Commands

Run the default SVM + BPSO workflow:

```bash
python -m hcc_radiomics.cli ^
  --data-path data/hcc_radiomics.xlsx ^
  --target-column Stage ^
  --method svm_bpso ^
  --output-dir outputs/run_001 ^
  --random-state 42
```

Run ANOVA F-test followed by BPSO:

```bash
python -m hcc_radiomics.cli ^
  --data-path data/hcc_radiomics.xlsx ^
  --target-column Stage ^
  --method svm_kbest_bpso ^
  --k-best 150 ^
  --output-dir outputs/run_002
```

Use a config file:

```bash
python -m hcc_radiomics.cli --config configs/default.yaml
```

## Methods

Available methods:

- `svm`: preprocessing plus SVM;
- `svm_kbest`: preprocessing, ANOVA `SelectKBest`, SVM;
- `svm_bpso`: preprocessing, BPSO feature selection, SVM;
- `svm_kbest_bpso`: preprocessing, ANOVA `SelectKBest`, BPSO feature selection, SVM.

`SelectKBest` validates `k` and clamps it to the available feature count with a warning. BPSO evaluates only selected feature masks; all-zero masks receive a penalty and are never treated as all features selected.

## Evaluation Metrics

The pipeline reports accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1-score, confusion matrix, selected feature count, ROC-AUC, and average precision where valid. ROC-AUC and precision-recall metrics use probabilities, not hard class labels. Multiclass targets are evaluated with macro summaries and one-vs-rest probability metrics where valid.

## Output Files

Each run creates the output directory automatically and writes:

- `config_used.json`;
- `dataset_summary.json`;
- `split_class_distributions.json`;
- `selected_features.txt`;
- `metrics.json`;
- `test_predictions.csv` with safe row indices;
- `confusion_matrix.png`;
- `roc_curve.png`;
- `precision_recall_curve.png`;
- `model.joblib`;
- `run.log`.

## Reproducibility

Set `--random-state` to control dataset splitting and BPSO randomness. The split code checks that train, validation, and test row indices do not overlap. The final test split is used only for evaluation.

## Testing

Tests use synthetic data only and do not require private research data.

```bash
python -m compileall .
pytest -q
```

## Known Limitations

- The real HCC dataset is not included and has not been validated in this branch.
- BPSO can be computationally expensive for hundreds of features; increase particles and iterations carefully.
- The project does not define a clinical stage-conversion rule. It uses the labels supplied in the target column.
- TPOT experiments from the historical scripts were not restored as active functionality because they were exploratory and leakage-prone.

## Troubleshooting

- If `Stage` is missing, pass the correct target name with `--target-column`.
- If repeated patient records exist, provide `--group-column`.
- If `SelectKBest` warns about `k`, reduce `--k-best` or confirm that clamping is acceptable.
- If ROC-AUC is unavailable, check whether the test split contains all classes and whether the model produced probabilities.

## Citation

No publication details or formal citation metadata were available in the recovered repository. Add manuscript, dataset, and software citations here when confirmed by the maintainers.

## Licence

No licence file was present in the recovered history. Reuse and redistribution rights are therefore unclear until the repository owner adds an explicit licence.

## Maintainer

Repository owner: `NSyaz/HCC_Radiomics`. Historical commits list GitHub user `NSyaz` as the uploader.
