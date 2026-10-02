# CoFi

**CoFi: Multi-View Contrastive Learning with Meta-Paths for Interpretable Drug–Target Interaction Prediction**

This repository contains the CoFi implementation and the three main experimental settings reported in the paper: Biased, Strategy-P, and Strategy-U.

## Installation

Python 3.10.16 and CUDA 12.1 were used in our experiments. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Quick start

1. Install the dependencies as described above. The raw DTINet benchmark files and the SMILES/FASTA inputs required for preprocessing are provided under `data/Luo/`.
2. From the repository root, generate all model inputs from the released source data:

```bash
python data/preprocessing/prepare_drug_target.py
python data/preprocessing/features/generate_drug_features.py
python data/preprocessing/features/generate_protein_features.py
python data/preprocessing/features/generate_disease_features.py
python data/preprocessing/generate_graph.py
python data/preprocessing/fine_view/generate_metapath_instances.py
```

These commands generate the labeled drug–target pairs, drug/protein/disease features, heterogeneous graph, node-type mask, and fine-grained meta-path instances required by the training code. See [`data/preprocessing/README.md`](data/preprocessing/README.md) for details. The fold-specific coarse-view matrices do not require a separate preprocessing command because they are generated automatically during cross-validation.

3. Run one of the three main experimental settings from the repository root:

```bash
# Biased setting with 1:1 positive-to-negative sampling
python main.py --mode biased --seed 2 --dropout_rate 0.2 --patience 5

# Strategy-P
python main.py --mode strategy_p --seed 2 --dropout_rate 0.2 --patience 5

# Strategy-U
python main.py --mode strategy_u --seed 2 --dropout_rate 0.2 --patience 5
```

All three settings use five-fold cross-validation, random seed 2, dropout 0.2, and early-stopping patience 5. Biased and Strategy-P use a learning rate of 0.0025; Strategy-U uses a learning rate of 0.004. These learning rates are selected automatically by the experiment mode.

For every fold, the coarse-grained adjacency matrices are generated from all auxiliary edges and that fold's training-positive DTI edges only. Validation- and test-positive DTI edges are excluded. The generated matrices are cached under `data/fold_specific_coarse/` and reused only when the training-positive split hash matches.

## Data description

The repository includes the source files required for preprocessing under `data/Luo/`. Running the preprocessing commands in the Quick start section generates the following model inputs under `data/`:

- `adjlists_idx/`: meta-path instances, adjacency lists, and instance-index files used by the fine-grained view.
- `features/drug.npy`: generated drug feature matrix.
- `features/protein.csv`: generated protein feature matrix.
- `features/disease.npy`: generated disease feature matrix.
- `adjM.npz`: heterogeneous-network adjacency matrix.
- `type_mask.npy`: node-type index array.
- `drug_target_offset.csv`: drug–target pairs and interaction labels used for training and evaluation.

The fold-specific coarse-view matrices are generated automatically at runtime and are not included in the repository.

The source networks under `data/Luo/` originate from the [DTINet repository](https://github.com/luoyunan/DTINet). The SMILES and FASTA inputs used for feature extraction are included in the same directory. Third-party data remain subject to their original licenses and terms of use.

## Reference data splits

The five-fold splits used for the reported experiments are provided for transparency and verification under:

```text
splits/
├── biased/
├── strategy_p/
└── strategy_u/
```

Each setting contains the corresponding training, validation, and test pairs for the five folds. These reference files are not loaded directly by the main training commands. During reproduction, each experimental mode deterministically constructs its splits from `data/drug_target_offset.csv` using the random seed specified on the command line. Using `--seed 2` follows the split rule used for the reported experiments, while other seed values generate the corresponding alternative splits. Splits generated during a run may be saved separately under a seed-specific path such as `splits/<setting>/seed_<seed>/fold_<fold>/`.

## Repository structure

```text
CoFi/
├── main.py
├── models/
├── train/
├── evaluation/
├── utils/
├── data/
├── splits/
└── supplementary/
```

- `models/`: CoFi model components and loss functions.
- `train/`: training implementations for the three main settings.
- `evaluation/`: evaluation metrics and saved-model evaluation utilities.
- `utils/`: data loading, sampling, split, and shared experiment utilities.
- `supplementary/hyperparameter_search/`: the ten evaluated hyperparameter combinations and validation summaries.
- `supplementary/temperature_sensitivity/`: temperature-sensitivity results reported in the response.
- `supplementary/pathway_interpretability/`: aggregate pathway-interpretability results reported in the manuscript and response.

The repository does not include epoch-level hyperparameter-search logs, per-prediction pathway CSV files, intermediate experiment outputs, or full training logs.
