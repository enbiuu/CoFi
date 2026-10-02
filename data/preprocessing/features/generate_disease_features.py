"""Generate 128-dimensional disease Node2vec features."""

import argparse
import csv
from pathlib import Path

import networkx as nx
import numpy as np
from gensim.models import Word2Vec


DATA_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = DATA_DIR / "Luo"
DEFAULT_OUTPUT = DATA_DIR / "features" / "disease.npy"
NUM_DISEASES = 5603


def read_positive_edges(path):
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if len(row) < 2:
                raise ValueError(f"Expected at least two columns in {path} at line {line_number}")
            if len(row) >= 3 and float(row[2]) <= 0:
                continue
            yield int(row[0]), int(row[1])


def build_graph():
    graph = nx.Graph()
    for drug_id, disease_id in read_positive_edges(RAW_DIR / "drug_dis.dat"):
        graph.add_edge(f"drug_{drug_id}", f"disease_{disease_id}", weight=1.0)
    for protein_id, disease_id in read_positive_edges(RAW_DIR / "protein_dis.dat"):
        graph.add_edge(f"protein_{protein_id}", f"disease_{disease_id}", weight=1.0)
    return graph


def simulate_walks(graph, walk_length, num_walks, rng):
    nodes = sorted(graph.nodes())
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_length:
                neighbors = sorted(graph.neighbors(walk[-1]))
                if not neighbors:
                    break
                weights = np.asarray([graph[walk[-1]][neighbor]["weight"] for neighbor in neighbors], dtype=float)
                probabilities = weights / weights.sum()
                walk.append(neighbors[rng.choice(len(neighbors), p=probabilities)])
            walks.append(walk)
    return walks


def generate_features(output_path, dimensions=128, walk_length=10, num_walks=100, seed=2):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    graph = build_graph()
    walks = simulate_walks(graph, walk_length, num_walks, rng)
    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=10,
        min_count=0,
        sg=1,
        workers=1,
        seed=seed,
    )

    matrix = rng.normal(scale=0.1, size=(NUM_DISEASES, dimensions)).astype(np.float32)
    covered = 0
    for disease_id in range(NUM_DISEASES):
        key = f"disease_{disease_id}"
        if key in model.wv:
            matrix[disease_id] = model.wv[key]
            covered += 1
    np.save(output_path, matrix)
    print(f"Saved {matrix.shape} disease feature matrix to {output_path}")
    print(f"Embedded disease nodes: {covered:,}; initialized missing nodes: {NUM_DISEASES - covered:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--walk-length", type=int, default=10)
    parser.add_argument("--num-walks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    generate_features(args.output, args.dimensions, args.walk_length, args.num_walks, args.seed)


if __name__ == "__main__":
    main()
