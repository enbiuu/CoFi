# Hyperparameter Search for Strategy-P

This directory reports the ten candidate hyperparameter configurations evaluated for Strategy-P and the configuration selected using five-fold validation AUPR.

## Hyperparameters

The reported symbols correspond to the implementation as follows:

| Symbol | Description | Configuration key |
|---|---|---|
| F | Output/embedding dimension | `out_dim` |
| H | Attention-vector dimension | `attn_vec_dim` |
| D | Hidden-layer dimension | `hidden_dim` |
| C | Number of attention heads | `num_heads` |
| K | Number of neighbor nodes sampled per meta-path | `neighbor_samples` |

The exact ten tested combinations are listed in `hyperparameter_configurations.csv`.

## Selection protocol

For each candidate configuration, five folds (`fold_0` to `fold_4`) were evaluated. Within each fold, the highest **validation AUPR** across training epochs was retained. The five retained values were then summarized using their arithmetic mean and population standard deviation (`ddof = 0`).

The configuration with the largest mean validation AUPR was selected. Test-set results were not used for hyperparameter selection.

The selected Strategy-P configuration is `trial_8`: F = 24, H = 24, D = 48, C = 1, and K = 100. Its five-fold validation AUPR is **0.698003 +/- 0.027178**.

## Files

- `hyperparameter_configurations.csv`: the ten tested F/H/D/C/K combinations.
- `hyperparameter_search_results.csv`: fold-level best validation AUPR and the mean/standard deviation for every trial.

