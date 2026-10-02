"""Create the global-ID DTI file used by the training code."""

import argparse
import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DATA_DIR / "Luo" / "drug_target.dat"
DEFAULT_OUTPUT = DATA_DIR / "drug_target_offset.csv"


def prepare_drug_target(input_path, output_path, protein_offset=708):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    positive_count = 0
    protein_min = None
    protein_max = None

    with input_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.reader(source)
        writer = csv.writer(destination)
        writer.writerow(["drug_id", "protein_id", "interaction"])

        for line_number, row in enumerate(reader, start=1):
            if len(row) != 3:
                raise ValueError(f"Expected 3 columns at line {line_number}: {row}")
            drug_id, protein_id, interaction = map(int, row)
            global_protein_id = protein_id + protein_offset
            writer.writerow([drug_id, global_protein_id, interaction])

            row_count += 1
            positive_count += int(interaction == 1)
            protein_min = global_protein_id if protein_min is None else min(protein_min, global_protein_id)
            protein_max = global_protein_id if protein_max is None else max(protein_max, global_protein_id)

    negative_count = row_count - positive_count
    if protein_min != 708 or protein_max != 2219:
        raise ValueError(f"Unexpected global protein-ID range: {protein_min}–{protein_max}")

    print(f"Saved {row_count:,} rows to {output_path}")
    print(f"Positive: {positive_count:,}; negative: {negative_count:,}")
    print(f"Global protein-ID range: {protein_min}–{protein_max}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protein-offset", type=int, default=708)
    args = parser.parse_args()
    prepare_drug_target(args.input, args.output, args.protein_offset)


if __name__ == "__main__":
    main()
