# Temperature Sensitivity

This supplementary result summarizes the sensitivity of CoFi to the temperature parameter used in the contrastive objective. The experiment was conducted under the Biased setting with five-fold cross-validation. All values are reported as mean ± standard deviation across the five folds.

## Aggregate results

| Temperature τ | AUC | AUPR |
|---:|---:|---:|
| 0.1 | 0.961 ± 0.008 | 0.963 ± 0.008 |
| 0.2 | 0.964 ± 0.003 | 0.968 ± 0.004 |
| 0.5 | 0.962 ± 0.009 | 0.964 ± 0.010 |
| 1.0 | **0.979 ± 0.001** | **0.973 ± 0.009** |
| 2.0 | 0.965 ± 0.007 | 0.968 ± 0.007 |

Performance is stable across moderate temperature values and reaches its highest AUC and AUPR at τ = 1.0. Accordingly, τ = 1.0 is used as the default setting in the reported CoFi experiments.

Only the aggregate results reported in the response are provided here. Fold-level outputs and full training logs are not included.
