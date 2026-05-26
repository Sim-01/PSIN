# Data availability

This public repository contains a synthetic seismic example and scripts that generate the example from scratch. The synthetic data are sufficient to verify the PSIN implementation and reproduce the computational workflow.

Some field datasets used in the manuscript may be subject to data-use or redistribution restrictions. These field data are therefore not included in this public repository unless the authors have explicit redistribution permission. Users who have legal access to a field dataset can place the files under:

```text
data/private/
```

This directory is ignored by Git to avoid accidental redistribution of restricted data.

Supported input formats include:

- `.npy`
- `.npz`
- `.mat`
- raw `.dat` with a user-provided shape

Example commands are provided in the main `README.md` and in `docs/tutorial.md`.

If a field-data result in the manuscript cannot be exactly reproduced because the corresponding data are restricted, reviewers can still run the bundled synthetic example and the field-data scripts on any local seismic section with the same interface.
