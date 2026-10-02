"""Generate 128-bit ECFP4/Morgan fingerprints from the released SMILES file."""

import argparse
import csv
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


DATA_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = DATA_DIR / "Luo" / "drug_smiles.csv"
DEFAULT_OUTPUT = DATA_DIR / "features" / "drug.npy"


def generate_fingerprints(input_path, output_path, radius=2, n_bits=128):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    drug_ids = []
    fingerprints = []
    with input_path.open("r", encoding="utf-8", newline="") as source:
        for row_number, row in enumerate(csv.reader(source), start=1):
            if len(row) < 2:
                raise ValueError(f"Expected DrugBank ID and SMILES at line {row_number}")
            drug_id, smiles = row[0].strip(), row[1].strip()
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Invalid SMILES for {drug_id} at line {row_number}")
            fingerprint = generator.GetFingerprint(molecule)
            array = np.zeros((n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fingerprint, array)
            drug_ids.append(drug_id)
            fingerprints.append(array)

    matrix = np.stack(fingerprints)
    if matrix.shape != (708, n_bits):
        raise ValueError(f"Unexpected drug feature shape: {matrix.shape}; expected (708, {n_bits})")
    np.save(output_path, matrix)
    print(f"Saved {matrix.shape} ECFP4 feature matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=128)
    args = parser.parse_args()
    generate_fingerprints(args.input, args.output, args.radius, args.n_bits)


if __name__ == "__main__":
    main()
