"""Generate the standard 147-dimensional CTD protein descriptors."""

import argparse
import csv
import math
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = DATA_DIR / "Luo" / "protein_fasta.csv"
DEFAULT_OUTPUT = DATA_DIR / "features" / "protein.csv"


# Three amino-acid groups for each of the seven standard CTD properties.
CTD_GROUPS = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "vdw_volume": ("GASCTPD", "NVEQIL", "MHKFRYW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary_structure": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solvent_accessibility": ("ALFCGIVW", "RKQEND", "MPSTHY"),
}


def group_sequence(sequence, groups):
    mapping = {amino_acid: group_id for group_id, group in enumerate(groups, start=1) for amino_acid in group}
    # Selenocysteine (U) is assigned to the same group as cysteine.
    mapping["U"] = mapping["C"]
    unknown = sorted(set(sequence) - set(mapping))
    if unknown:
        raise ValueError(f"Unsupported amino-acid symbols: {unknown}")
    return [mapping[amino_acid] for amino_acid in sequence]


def composition(grouped):
    length = len(grouped)
    return [grouped.count(group_id) / length for group_id in (1, 2, 3)]


def transition(grouped):
    if len(grouped) == 1:
        return [0.0, 0.0, 0.0]
    pairs = ((1, 2), (1, 3), (2, 3))
    denominator = len(grouped) - 1
    return [
        sum({left, right} == set(pair) for left, right in zip(grouped, grouped[1:])) / denominator
        for pair in pairs
    ]


def distribution(grouped):
    length = len(grouped)
    values = []
    for group_id in (1, 2, 3):
        positions = [index for index, value in enumerate(grouped, start=1) if value == group_id]
        if not positions:
            values.extend([0.0] * 5)
            continue
        selected = [
            positions[0],
            positions[math.ceil(0.25 * len(positions)) - 1],
            positions[math.ceil(0.50 * len(positions)) - 1],
            positions[math.ceil(0.75 * len(positions)) - 1],
            positions[-1],
        ]
        values.extend(position / length * 100.0 for position in selected)
    return values


def feature_names():
    names = []
    percentiles = ("first", "25pct", "50pct", "75pct", "last")
    for property_name in CTD_GROUPS:
        names.extend(f"{property_name}_composition_g{group_id}" for group_id in (1, 2, 3))
        names.extend(f"{property_name}_transition_{pair}" for pair in ("12", "13", "23"))
        names.extend(
            f"{property_name}_distribution_g{group_id}_{percentile}"
            for group_id in (1, 2, 3)
            for percentile in percentiles
        )
    if len(names) != 147:
        raise AssertionError(f"Expected 147 feature names, found {len(names)}")
    return names


def encode_sequence(sequence):
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("Protein sequence is empty")
    features = []
    for groups in CTD_GROUPS.values():
        grouped = group_sequence(sequence, groups)
        features.extend(composition(grouped))
        features.extend(transition(grouped))
        features.extend(distribution(grouped))
    if len(features) != 147:
        raise AssertionError(f"Expected 147 CTD features, found {len(features)}")
    return features


def generate_features(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if len(row) < 2:
                raise ValueError(f"Expected protein ID and sequence at line {line_number}")
            protein_id, sequence = row[0].strip(), row[1].strip()
            rows.append([protein_id, *encode_sequence(sequence)])

    if len(rows) != 1512:
        raise ValueError(f"Unexpected protein count: {len(rows)}; expected 1512")

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["protein_id", *feature_names()])
        writer.writerows(rows)
    print(f"Saved 1,512 × 147 CTD feature matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate_features(args.input, args.output)


if __name__ == "__main__":
    main()
