# Planned Real-Data Experiment

This document describes the intended real-data workflow. It is not a report of completed results.

1. Validate dataset schema and confirm that `Stage` uses the mapping `0 = groups/stages 1-2` and `1 = groups/stages 3-4`.
2. Inspect class balance, missing values, duplicate rows, constant features, and metadata columns.
3. Run the `dummy` classifier baseline.
4. Run `svm`.
5. Run `svm_kbest`.
6. Run `svm_bpso`.
7. Run `svm_kbest_bpso`.
8. Repeat evaluation across multiple random seeds or repeated stratified folds.
9. Compare feature counts, predictive performance, variance across repeats, and feature-selection stability.
10. Produce final tables and figures only after real-data validation is complete.

Single split results should be treated as exploratory. Final scientific conclusions should include uncertainty and avoid claiming model performance from one split alone.
