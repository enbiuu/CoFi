# CoFi
CoFi: Multi-View Contrastive Learning with Meta-Paths for Interpretable Drug–Target Interaction Prediction

# Requirements
The code has been tested with the following environment:
* Python 3.10.16
* numpy 1.24.3
* PyTorch 2.4.0
* scikit-learn 1.0.2
* DGL 2.0.0.cu121

# Quick start
To reproduce our results:
1. Unzip data.zip in data/ directory.
2. Choose the experimental setting:
* Run main_biased.py to reproduce the cross-validation results under the 1:1 sampling setting.
* Run main_Strategy_P.py to reproduce the cross-validation results under Strategy P.
* Run main_Strategy_U.py to reproduce the cross-validation results under Strategy U.

# Data description
* adjlists_idx/:Meta-path instances and corresponding adjacency lists used for the Fine-grained view.
* adjs/:Meta-path–induced adjacency matrices used for the Coarse-grained view.
* features/:Pre-extracted feature matrices for drugs、proteins、diseases
* drug_target.csv：drug–target interaction pairs used for training and evaluation.
* drug.txt: list of drug names.
* protein.txt: list of protein names.
* disease.txt: list of disease names.
* protein_protein.dat : Protein-Protein interaction matrix.
* drug_drug.dat : Drug-Drug interaction matrix.
* protein_disease.dat : Protein-Disease association matrix.
* drug_disease.dat : Drug-Disease association matrix.

These files: drug.txt, protein.txt, disease.txt, protein_protein.dat, drug_drug.dat, protein_disease.dat, drug_disease.dat, are extracted from https://github.com/luoyunan/DTINet.


