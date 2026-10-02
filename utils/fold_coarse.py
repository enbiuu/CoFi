"""Generate and load leakage-free coarse-view matrices for one CV fold."""

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from utils.data import load_normalized_adj
from utils.experiment_common import load_mean_adjs


NUM_DRUGS = 708
PROTEIN_OFFSET = NUM_DRUGS
PROTEIN_END = NUM_DRUGS + 1512

METAPATHS = (
    (0, 0, 0, 1),
    (0, 1, 1, 1),
    (0, 2, 0, 1),
    (1, 0, 2, 0, 1),
    (1, 0, 1),
)


def _canonical_pairs(train_pos):
    if hasattr(train_pos, "detach"):
        train_pos = train_pos.detach().cpu().numpy()
    pairs = np.asarray(train_pos, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("train_pos must have shape (n, 2)")
    pairs = np.unique(pairs, axis=0)
    if len(pairs) == 0:
        raise ValueError("train_pos is empty")
    if np.any((pairs[:, 0] < 0) | (pairs[:, 0] >= NUM_DRUGS)):
        raise ValueError("Drug IDs must be in the global range 0-707")
    if np.any((pairs[:, 1] < PROTEIN_OFFSET) | (pairs[:, 1] >= PROTEIN_END)):
        raise ValueError("Protein IDs must be in the global range 708-2219")
    return pairs


def _training_hash(train_pos):
    pairs = _canonical_pairs(train_pos).astype("<i8", copy=False)
    return hashlib.sha256(pairs.tobytes()).hexdigest()


def _fold_graph(full_adjacency, type_mask, train_pos):
    """Keep every auxiliary edge and only the fold's training-positive DTIs."""
    pairs = _canonical_pairs(train_pos)
    coo = full_adjacency.tocoo()
    is_dti = (
        ((type_mask[coo.row] == 0) & (type_mask[coo.col] == 1))
        | ((type_mask[coo.row] == 1) & (type_mask[coo.col] == 0))
    )
    rows = np.concatenate((coo.row[~is_dti], pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((coo.col[~is_dti], pairs[:, 1], pairs[:, 0]))
    graph = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=full_adjacency.shape,
    )
    graph.data[:] = 1.0
    graph.eliminate_zeros()
    return graph


def _relation_matrix(adjacency, type_mask, source_type, target_type):
    coo = adjacency.tocoo()
    keep = (type_mask[coo.row] == source_type) & (type_mask[coo.col] == target_type)
    return sp.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float64), (coo.row[keep], coo.col[keep])),
        shape=adjacency.shape,
    )


def _count_instances(relations, metapath):
    counts = relations[(metapath[0], metapath[1])]
    for source_type, target_type in zip(metapath[1:-1], metapath[2:]):
        counts = counts @ relations[(source_type, target_type)]
        counts.eliminate_zeros()
    if metapath[0] != metapath[-1]:
        counts = counts + counts.T
    else:
        counts = counts.maximum(counts.T)
    counts.eliminate_zeros()
    return counts.tocsr()


def _normalize(adjacency):
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt = np.zeros_like(degrees, dtype=np.float64)
    nonzero = degrees > 0
    inverse_sqrt[nonzero] = np.power(degrees[nonzero], -0.5)
    degree_matrix = sp.diags(inverse_sqrt, format="csr")
    normalized = degree_matrix @ adjacency @ degree_matrix
    normalized.eliminate_zeros()
    return normalized.astype(np.float32).tocsr()


def _matrix_paths(output_dir):
    paths = {}
    for metapath in METAPATHS:
        name = "-".join(map(str, metapath))
        paths[metapath] = output_dir / str(metapath[0]) / f"normalized_adj_{name}.npz"
    return paths


def ensure_fold_coarse_matrices(
    full_adjacency,
    type_mask,
    train_pos,
    data_dir,
    seed,
    fold_idx,
):
    """Generate missing/stale matrices and return their paths by meta-path."""
    data_dir = Path(data_dir)
    output_dir = data_dir / "fold_specific_coarse" / f"seed_{seed}" / f"fold_{fold_idx}"
    manifest_path = output_dir / "manifest.json"
    paths = _matrix_paths(output_dir)
    expected_hash = _training_hash(train_pos)

    manifest = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as source:
            manifest = json.load(source)
    cache_valid = (
        manifest is not None
        and manifest.get("train_pos_sha256") == expected_hash
        and manifest.get("metapaths") == [list(path) for path in METAPATHS]
        and all(path.exists() for path in paths.values())
    )
    if cache_valid:
        print(f"Reusing fold-specific coarse matrices: {output_dir}")
        return paths

    print(f"Generating fold-specific coarse matrices: seed={seed}, fold={fold_idx}")
    fold_graph = _fold_graph(full_adjacency, type_mask, train_pos)
    relation_types = {
        pair for metapath in METAPATHS for pair in zip(metapath[:-1], metapath[1:])
    }
    relations = {
        pair: _relation_matrix(fold_graph, type_mask, *pair) for pair in relation_types
    }

    nonzero_counts = {}
    for metapath in METAPATHS:
        counts = _count_instances(relations, metapath)
        normalized = _normalize(counts)
        output_path = paths[metapath]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(output_path, normalized)
        name = "-".join(map(str, metapath))
        nonzero_counts[name] = int(counts.nnz)
        print(f"  saved {name}: {counts.nnz:,} nonzero endpoint pairs")

    manifest = {
        "seed": int(seed),
        "fold": int(fold_idx),
        "training_positive_count": int(len(_canonical_pairs(train_pos))),
        "train_pos_sha256": expected_hash,
        "metapaths": [list(path) for path in METAPATHS],
        "nonzero_endpoint_pairs": nonzero_counts,
        "graph_policy": "all auxiliary edges plus fold training-positive DTI edges only",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2)
    return paths


def load_fold_coarse_adjs(
    full_adjacency,
    type_mask,
    train_pos,
    data_dir,
    seed,
    fold_idx,
    device,
):
    paths = ensure_fold_coarse_matrices(
        full_adjacency,
        type_mask,
        train_pos,
        data_dir,
        seed,
        fold_idx,
    )
    drug_paths = [str(paths[path]) for path in METAPATHS if path[0] == 0]
    target_paths = [str(paths[path]) for path in METAPATHS if path[0] == 1]
    return load_mean_adjs(load_normalized_adj, drug_paths, target_paths, device)
