# Data Directory

Place authorised local research data in this directory. Data files are ignored by Git because the recovered historical workbooks contain row-level HCC radiomics and TCGA-style patient/sample information.

## Expected Analysis Dataset

The main command expects a tabular `.xlsx`, `.csv`, or `.tsv` file with:

- one target column, typically `Stage`;
- numeric radiomics feature columns;
- optional identifier or metadata columns such as `patient_id`, `case_id`, `record_id`, or acquisition metadata.

Identifier and metadata columns should be passed with `--metadata-column` and, when repeated patient records exist, `--group-column` so grouped splitting can keep a patient out of multiple splits.

## Privacy

Do not commit patient identifiers, protected health information, accessions, dates of birth, clinical free text, credentials, local machine paths, or private institutional metadata. Authorised researchers should provide the dataset locally after confirming governance and ethics requirements.

## Historical Files

The original commit `33aa78fd27bee2ec88f34037beaf104f39fcc64c` contained:

- `Pycharm copy.xlsx`: a 40-row by 663-column stage-labelled radiomics table with binary `Stage` values in the inspected sheet.
- `LiverHccSeg (Open Source).xlsx`: multiple TCGA-style patient/sample sheets and segmentation feature sheets.

Both files were deliberately excluded from the rebuilt repository because they are data-bearing workbooks.
