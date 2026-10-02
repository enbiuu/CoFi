"""Generate normalized meta-path adjacency matrices for CoFi's coarse view."""

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp


DATA_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = DATA_DIR / "adjM.npz"
DEFAULT_TYPE_MASK = DATA_DIR / "type_mask.npy"
DEFAULT_OUTPUT = DATA_DIR / "adjs_no_pathsim"

# These filenames are the inputs loaded by the three main experiment scripts.
METAPATHS = (
    (0, 0, 0, 1),
    (0, 1, 1, 1),
    (0, 2, 0, 1),
    (1, 0, 2, 0, 1),
    (1, 0, 1),
)


def relation_matrix(adjacency, type_mask, source_type, target_type):
    coo = adjacency.tocoo()
    keep = (type_mask[coo.row] == source_type) & (type_mask[coo.col] == target_type)
    data = np.ones(int(keep.sum()), dtype=np.float64)
    return sp.csr_matrix(
        (data, (coo.row[keep], coo.col[keep])), shape=adjacency.shape
    )


def count_metapath_instances(relations, metapath):
    counts = relations[(metapath[0], metapath[1])]
    for source_type, target_type in zip(metapath[1:-1], metapath[2:]):
        counts = counts @ relations[(source_type, target_type)]
        counts.eliminate_zeros()

    # Cross-type paths are directed. Accumulating the reverse counts produces
    # the symmetric full-graph adjacency described in the manuscript.
    if metapath[0] != metapath[-1]:
        counts = counts + counts.T
    else:
        # Palindromic same-type paths are symmetric by construction. This
        # guards against any storage asymmetry without doubling their counts.
        counts = counts.maximum(counts.T)
    counts.eliminate_zeros()
    return counts.tocsr()


def symmetric_normalize(adjacency):
    """Compute D^(-1/2) A D^(-1/2), matching manuscript Eq. (2)."""
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degrees, dtype=np.float64)
    nonzero = degrees > 0
    inverse_sqrt[nonzero] = np.power(degrees[nonzero], -0.5)
    degree_matrix = sp.diags(inverse_sqrt, format="csr")
    normalized = degree_matrix @ adjacency @ degree_matrix
    normalized.eliminate_zeros()
    return normalized.astype(np.float32).tocsr()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--type-mask", type=Path, default=DEFAULT_TYPE_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    adjacency = sp.load_npz(args.graph).tocsr()
    type_mask = np.load(args.type_mask)
    if adjacency.shape[0] != len(type_mask):
        raise ValueError("The graph and type mask contain different numbers of nodes")

    relation_types = {
        pair for metapath in METAPATHS for pair in zip(metapath[:-1], metapath[1:])
    }
    relations = {
        pair: relation_matrix(adjacency, type_mask, *pair) for pair in relation_types
    }

    for metapath in METAPATHS:
        counts = count_metapath_instances(relations, metapath)
        normalized = symmetric_normalize(counts)
        output_dir = args.output_dir / str(metapath[0])
        output_dir.mkdir(parents=True, exist_ok=True)
        name = "-".join(map(str, metapath))
        output_path = output_dir / f"normalized_adj_{name}.npz"
        sp.save_npz(output_path, normalized)
        print(f"Saved {output_path} ({counts.nnz:,} nonzero endpoint pairs)")


if __name__ == "__main__":
    main()
