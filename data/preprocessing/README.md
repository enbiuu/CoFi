# Data preprocessing

All commands below are run from the repository root. The scripts use paths relative to the repository and do not require machine-specific path changes.

## Drug–target pairs

Create the global-ID interaction file used by the training code. Protein IDs are shifted by 708, producing the global range 708–2219.

```bash
python data/preprocessing/prepare_drug_target.py
```

## Drug features

Generate 128-bit Morgan fingerprints (ECFP4; radius 2) from the SMILES strings in `data/Luo/drug_smiles.csv`.

```bash
python data/preprocessing/features/generate_drug_features.py
```

The output is written to `data/features/drug.npy` with shape `(708, 128)`.

## Protein features

Generate the standard 147-dimensional Composition, Transition, Distribution (CTD) descriptors from the amino-acid sequences in `data/Luo/protein_fasta.csv`.

```bash
python data/preprocessing/features/generate_protein_features.py
```

The output is written to `data/features/protein.csv`. Its first column contains the protein identifier, followed by 147 CTD feature columns. Selenocysteine (`U`) is assigned to the same CTD property group as cysteine (`C`).

## Disease features

Generate 128-dimensional Node2vec features from the drug–disease and protein–disease association networks. A fixed seed is used, and disease nodes absent from both networks receive deterministic random initializations.

```bash
python data/preprocessing/features/generate_disease_features.py
```

The output is written to `data/features/disease.npy` with shape `(5603, 128)`.

## Heterogeneous graph

Build `data/adjM.npz` and `data/type_mask.npy` from the five relation files in `data/Luo/`.

```bash
python data/preprocessing/generate_graph.py
```

## Fine-grained view

Generate the complete meta-path instances used by MAGNN. The script processes the five retained meta-paths used in the main experiments and writes `.adjlist`, `.json`, and `_idx.pickle` files under `data/adjlists_idx/`.

```bash
python data/preprocessing/fine_view/generate_metapath_instances.py
```

The generated instance arrays preserve all intermediate nodes and are stored in the target-to-source order expected by the model. The separate conversion utility is only needed when converting existing `.adjlist` files that do not already have matching JSON files:

```bash
python data/preprocessing/fine_view/convert_adjlists_to_json.py
```

## Coarse-grained view

The main experiment commands automatically generate one set of five coarse-view matrices for each cross-validation fold. Before constructing a fold's matrices, all DTI edges are removed from the heterogeneous graph and only that fold's training-positive DTI edges are added back. Validation- and test-positive DTI edges are therefore excluded.

For every retained meta-path, the matrix entries contain meta-path instance counts. Forward and reverse counts are accumulated for directed cross-type paths, followed by the symmetric normalization

```text
D^(-1/2) A D^(-1/2).
```

The matrices are cached under:

```text
data/fold_specific_coarse/seed_<seed>/fold_<fold>/
```

The cache includes a hash of the training-positive split and is regenerated automatically if the split changes. The model averages the three drug-side matrices and the two protein-side matrices at load time. No PathSim filtering or top-k pruning is applied.

The standalone command below generates matrices from the complete graph and is retained as a preprocessing utility; it is not used by the main cross-validation experiments:

```bash
python data/preprocessing/coarse_view/generate_metapath_adjacencies.py
```

Run `generate_graph.py` before either view-specific preprocessing command.
