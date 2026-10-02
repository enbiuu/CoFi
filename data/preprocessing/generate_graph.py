"""Build the heterogeneous adjacency matrix and node-type mask."""

import csv
from pathlib import Path

import numpy as np
from scipy.sparse import dok_matrix, save_npz


DATA_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = DATA_DIR / "Luo"
NUM_DRUGS = 708
NUM_PROTEINS = 1512
NUM_DISEASES = 5603
PROTEIN_OFFSET = NUM_DRUGS
DISEASE_OFFSET = NUM_DRUGS + NUM_PROTEINS


def read_positive_edges(path):
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if len(row) < 2:
                raise ValueError(f"Expected at least two columns in {path} at line {line_number}")
            if len(row) >= 3 and float(row[2]) <= 0:
                continue
            yield int(row[0]), int(row[1])


def add_undirected_edges(matrix, edges, left_offset=0, right_offset=0):
    for left, right in edges:
        left += left_offset
        right += right_offset
        matrix[left, right] = 1
        matrix[right, left] = 1


def main():
    total_nodes = NUM_DRUGS + NUM_PROTEINS + NUM_DISEASES
    adjacency = dok_matrix((total_nodes, total_nodes), dtype=np.uint8)

    add_undirected_edges(adjacency, read_positive_edges(RAW_DIR / "drug_target.dat"), 0, PROTEIN_OFFSET)
    add_undirected_edges(adjacency, read_positive_edges(RAW_DIR / "drug_dis.dat"), 0, DISEASE_OFFSET)
    add_undirected_edges(adjacency, read_positive_edges(RAW_DIR / "protein_dis.dat"), PROTEIN_OFFSET, DISEASE_OFFSET)
    add_undirected_edges(adjacency, read_positive_edges(RAW_DIR / "drug_drug.dat"))
    add_undirected_edges(adjacency, read_positive_edges(RAW_DIR / "pro_pro.dat"), PROTEIN_OFFSET, PROTEIN_OFFSET)

    type_mask = np.zeros(total_nodes, dtype=np.int64)
    type_mask[PROTEIN_OFFSET:DISEASE_OFFSET] = 1
    type_mask[DISEASE_OFFSET:] = 2

    save_npz(DATA_DIR / "adjM.npz", adjacency.tocsr())
    np.save(DATA_DIR / "type_mask.npy", type_mask)
    print(f"Saved adjacency matrix with shape {adjacency.shape} and {adjacency.nnz} directed entries")


if __name__ == "__main__":
    main()
