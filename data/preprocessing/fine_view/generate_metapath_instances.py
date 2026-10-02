"""Generate the meta-path instances consumed by CoFi's fine-grained view."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


DATA_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = DATA_DIR / "adjM.npz"
DEFAULT_TYPE_MASK = DATA_DIR / "type_mask.npy"
DEFAULT_OUTPUT = DATA_DIR / "adjlists_idx"

# The five retained meta-paths used by all three main experimental settings.
METAPATHS = (
    (0, 0, 0, 1),
    (0, 1, 1, 1),
    (0, 2, 0, 1),
    (1, 0, 1),
    (1, 0, 2, 0, 1),
)


def enumerate_instances(adjacency, type_mask, start_node, metapath):
    """Return terminal neighbors and complete instances starting at one node."""
    frontier = [(start_node, (start_node,))]
    for required_type in metapath[1:]:
        next_frontier = []
        for current_node, path in frontier:
            begin = adjacency.indptr[current_node]
            end = adjacency.indptr[current_node + 1]
            for neighbor in adjacency.indices[begin:end]:
                if type_mask[neighbor] == required_type:
                    next_frontier.append((int(neighbor), path + (int(neighbor),)))
        frontier = next_frontier
        if not frontier:
            break

    neighbors = [terminal for terminal, _ in frontier]
    # MAGNN expects every stored instance in target-to-source order.
    instances = np.asarray([path[::-1] for _, path in frontier], dtype=np.int64)
    if instances.size == 0:
        instances = np.empty((0, len(metapath)), dtype=np.int64)
    return neighbors, instances


def generate_metapath(adjacency, type_mask, metapath, output_root):
    start_type = metapath[0]
    start_nodes = np.flatnonzero(type_mask == start_type)
    output_dir = output_root / str(start_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "-".join(map(str, metapath))

    text_lines = []
    json_adjacency = {}
    instance_index = {}
    for node in start_nodes:
        node = int(node)
        neighbors, instances = enumerate_instances(adjacency, type_mask, node, metapath)
        text_lines.append(" ".join(map(str, [node, *neighbors])))
        json_adjacency[str(node)] = neighbors
        instance_index[node] = instances

    adjlist_path = output_dir / f"{name}.adjlist"
    json_path = output_dir / f"{name}.json"
    index_path = output_dir / f"{name}_idx.pickle"
    adjlist_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as destination:
        json.dump(json_adjacency, destination)
    with index_path.open("wb") as destination:
        pickle.dump(instance_index, destination, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {name}: {sum(len(v) for v in json_adjacency.values()):,} instances")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--type-mask", type=Path, default=DEFAULT_TYPE_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    adjacency = load_npz(args.graph).tocsr()
    adjacency.sort_indices()
    type_mask = np.load(args.type_mask)
    if adjacency.shape[0] != len(type_mask):
        raise ValueError("The graph and type mask contain different numbers of nodes")

    for metapath in METAPATHS:
        generate_metapath(adjacency, type_mask, metapath, args.output_dir)


if __name__ == "__main__":
    main()
