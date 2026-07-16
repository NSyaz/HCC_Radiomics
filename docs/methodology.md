# Methodology

## Target Definition

The binary target column is `Stage`:

- `0` = HCC groups/stages 1-2
- `1` = HCC groups/stages 3-4

The active workflow preserves this mapping and does not reinterpret the clinical target.

## Dataset Dimensions

The primary dataset contains approximately 40 patients and approximately 662 radiomics predictors, with one row per patient. This is a high-dimensional, low-sample-size setting.

## Preprocessing

The workflow validates the target column, excludes metadata columns from predictors, converts candidate predictors to numeric values, handles infinite values as missing, imputes missing numeric values with the training median, removes constant features, and standardises features for SVM-based modelling.

Imputation, zero-variance filtering, and scaling are fitted on training data only, then applied to validation and test data.

## Modelling Methods

The planned comparison includes:

- `dummy`: stratified dummy baseline.
- `svm`: SVM without feature selection.
- `svm_kbest`: ANOVA F-test / `SelectKBest` followed by SVM.
- `svm_bpso`: BPSO feature selection followed by SVM.
- `svm_kbest_bpso`: ANOVA F-test followed by BPSO and SVM.

## ANOVA F-test

`SelectKBest(f_classif)` ranks numeric predictors using an ANOVA F-test fitted on training data only. The selected feature count `k_best` is configurable and should be tuned inside training validation. If `k_best` exceeds the available feature count, the implementation clamps it with a warning.

## BPSO Objective

BPSO searches binary feature masks using cross-validated balanced accuracy on the training split. The objective balances predictive error and selected feature fraction. All-zero masks receive an explicit penalty and are not treated as selecting all features.

## SVM Classifier

The SVM uses class balancing and probability estimates so ROC-AUC and precision-recall metrics can be computed from probabilities rather than hard labels.

## Leakage Prevention

The final test set is never used to fit preprocessing, feature selection, BPSO, or model parameters. Split indices are checked for overlap. Optional grouped splitting is available for future datasets with repeated patient rows.

## Evaluation Metrics

The pipeline reports accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1-score, confusion matrix, ROC-AUC, average precision, and selected feature count where applicable.

## Validation Caution

With approximately 662 predictors and approximately 40 patients, model variance and overfitting risk are substantial. Final scientific results should include repeated validation, uncertainty estimates, and feature-selection stability analysis before drawing conclusions.
