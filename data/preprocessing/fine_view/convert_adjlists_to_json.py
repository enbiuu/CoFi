"""Convert CoFi .adjlist files to the JSON format used by the data loader."""

import argparse
import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = DATA_DIR / "adjlists_idx"


def convert_directory(root):
    files = sorted(Path(root).rglob("*.adjlist"))
    if not files:
        raise FileNotFoundError(f"No .adjlist files found under {root}")
    for input_path in files:
        adjacency = {}
        with input_path.open("r", encoding="utf-8") as source:
            for line in source:
                values = [int(value) for value in line.split()]
                if values:
                    adjacency[str(values[0])] = values[1:]
        output_path = input_path.with_suffix(".json")
        with output_path.open("w", encoding="utf-8") as destination:
            json.dump(adjacency, destination)
        print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    convert_directory(args.input_dir)


if __name__ == "__main__":
    main()
