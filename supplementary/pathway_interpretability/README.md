# Pathway Annotation Consistency

This supplementary result evaluates whether the pathways inferred for CoFi's top-ranked unknown drug–target interaction (DTI) predictions are consistent with existing biological annotations. The analysis uses KEGG and Reactome pathway annotations obtained through the December 2025 CTD snapshot.

## Evaluation protocol

For each DTI in CoFi's top-1,000 and top-2,000 ranked unknown predictions, the top three pathways induced by meta-path intermediate nodes were extracted. Endpoint annotations were excluded from pathway inference and were used only for evaluation, thereby avoiding circular validation.

- **Evaluable DTI:** a predicted interaction for which at least one pathway could be inferred and at least one KEGG or Reactome annotation was available for the drug or protein endpoint.
- **Matched DTI:** an evaluable interaction for which at least one of the top three inferred pathways matched an existing annotation of either endpoint.
- **Match@3:** the number of matched DTIs divided by the number of evaluable DTIs.
- **Drug Match@3:** the proportion matching a drug-side pathway annotation.
- **Protein Match@3:** the proportion matching a protein-side pathway annotation.

Degree-matched random prediction sets were used as controls. Each random result is summarized over three independent draws.

## Aggregate results

| Prediction set | Total predictions | Replicates | Evaluable DTI | Matched DTI | Match@3 | Drug Match@3 | Protein Match@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoFi Top-1000 | 1,000 | 1 | 982 | 833 | **84.83%** | 65.17% | 58.86% |
| CoFi Top-2000 | 2,000 | 1 | 1,935 | 1,688 | **87.24%** | 66.51% | 61.45% |
| Degree-matched random, Top-1000 | 1,000 | 3 | 966 | 571 | 59.13% ± 1.22% | — | — |
| Degree-matched random, Top-2000 | 2,000 | 3 | 1,912 | 1,317 | 68.88% ± 0.98% | — | — |

For the three-replicate random controls, the evaluable and matched DTI counts are reported as rounded aggregate counts, while Match@3 is reported as mean ± standard deviation across the three draws.

Among CoFi's top-1,000 predictions, 982 were evaluable and 833 achieved Match@3. Among the top-2,000 predictions, 1,935 were evaluable and 1,688 achieved Match@3. CoFi therefore obtained substantially higher pathway-annotation consistency than the corresponding degree-matched random controls.

This analysis measures consistency and biological plausibility relative to existing knowledge bases. It does not establish a causal mechanism, and unmatched predictions may still represent novel biological hypotheses because KEGG and Reactome annotations are incomplete.

Only the aggregate statistics and metric definitions reported in the manuscript and response are provided here. Per-prediction pathway alignments, random-draw pair sets, and detailed CSV outputs are not included.
