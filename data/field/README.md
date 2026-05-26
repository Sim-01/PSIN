# Field data directory

Place local field seismic sections here only if you have permission to use and redistribute them.

Recommended public-repository practice:

- keep restricted field data out of Git;
- place private field data under `data/private/`, which is ignored by `.gitignore`;
- use `scripts/train_field_data.py` with the appropriate file path and shape.

The repository already includes a synthetic example under `data/sample/` for reviewers to test the code without restricted field data.
