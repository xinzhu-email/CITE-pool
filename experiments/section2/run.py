"""Train the section-2 joint reference/query representation on prepared inputs."""

import argparse
from pathlib import Path

from citepool.representation import RepresentationConfig, learn_representation


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "section2"

# Choose one prepared input set: healthy (CD8-weighted final model), patient, or s2p2.
INPUTS = {
    "healthy": DATA / "healthy",
    "patient": DATA / "patient",
    "s2p2": DATA / "s2p2",
}


def train(name: str, output_dir: Path) -> None:
    """Read the prepared RNA, taxonomy and masked protein target, then train."""
    base = INPUTS[name]
    query_name = {"healthy": "H00054", "patient": "N00023", "s2p2": "s2p2"}[name]
    rna = base / f"reference_plus_{query_name}_query_rna.h5ad"
    protein = base / "reference_only_protein_targets.h5ad"
    if not rna.is_file() or not protein.is_file() or not (base / "taxonomy").is_dir():
        raise FileNotFoundError(f"Prepared section-2 inputs are missing under {base}; see experiments/README.md")
    learn_representation(
        RepresentationConfig(
            rna_h5ad=rna,
            taxonomy_dir=base / "taxonomy",
            protein_target_h5ad=protein,
            output_dir=output_dir,
            protein_ratio=5.0,
            classification_leaf_weights={"ALIGNED_0007": 4.0, "ALIGNED_0009": 4.0} if name == "healthy" else {},
            accelerator="cpu" if name == "healthy" else "gpu",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", choices=INPUTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    train(options.query, options.output)
