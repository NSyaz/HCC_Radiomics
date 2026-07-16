# Data Directory

Place the authorised local analysis dataset here:

```text
data/hcc_radiomics.xlsx
```

Accepted formats are `.xlsx`, `.xls`, `.csv`, `.tsv`, and tab-delimited `.txt`.

## Expected Dataset

- Approximately 40 patients.
- Approximately 662 radiomics predictors.
- One row per patient.
- Target column: `Stage`.
- Confirmed binary mapping:
  - `0` = HCC groups/stages 1-2
  - `1` = HCC groups/stages 3-4

All approved numeric columns other than `Stage` are treated as candidate predictors unless explicitly excluded as metadata.

## Metadata Columns

Use `--metadata-column` to exclude approved non-predictor columns:

```bash
python -m hcc_radiomics.cli --data-path data/hcc_radiomics.xlsx --target-column Stage --metadata-column scanner_model --method dummy
```

A patient or group column is not required for the current dataset because each row is a different patient. If a future dataset contains repeated rows per patient, pass `--group-column` so grouped splitting keeps a group out of multiple splits.

## Privacy And Governance

Data files are ignored by Git because the repository must not distribute real patient-level data. Do not commit Excel workbooks, CSV/TSV exports, row-level derived data, patient identifiers, accessions, dates of birth, clinical free text, credentials, private institutional metadata, or local machine paths.

Authorised researchers should provide the dataset locally only after confirming governance, ethics, and data-sharing requirements.

## Historical Files

Historical commit `33aa78fd27bee2ec88f34037beaf104f39fcc64c` contained:

- `Pycharm copy.xlsx`
- `LiverHccSeg (Open Source).xlsx`

These workbooks were deliberately excluded from the rebuilt branch because they are data-bearing files.

## Quick Validation Command

Run a dummy baseline to validate schema, splitting, and output generation without claiming real model performance:

```bash
python -m hcc_radiomics.cli --data-path data/hcc_radiomics.xlsx --target-column Stage --method dummy --output-dir outputs/data_check
```
