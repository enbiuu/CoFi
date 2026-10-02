# Data

## Contents

- `Luo/`: raw DTINet benchmark files and the SMILES/FASTA inputs used for feature extraction.
- `preprocessing/`: preprocessing scripts that use repository-relative paths.
- `drug_target_offset.csv`: labeled drug–target pairs in the global node-ID space used by the training code. Drug IDs occupy 0–707 and protein IDs occupy 708–2219.

The source `Luo/drug_target.dat` uses local protein IDs in the range 0–1511. Run the following command to regenerate the offset file:

```bash
python data/preprocessing/prepare_drug_target.py
```

See `preprocessing/README.md` for the currently released preprocessing steps. Large derived graph and feature artifacts are not included in this directory.
