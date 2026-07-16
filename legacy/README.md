# Legacy Scripts

These files were recovered from historical commit `33aa78fd27bee2ec88f34037beaf104f39fcc64c` for auditability only. They are not maintained entry points.

Known issues in the recovered scripts include hard-coded local paths, fitting feature selection or models on test data, training SVMs on test subsets, fixed feature counts, unclear BPSO mask handling, and metrics computed from inappropriate prediction objects.

Use the package entry point instead:

```bash
python -m hcc_radiomics.cli --data-path data/hcc_radiomics.xlsx --target-column Stage --method svm_bpso --output-dir outputs/run_001
```
