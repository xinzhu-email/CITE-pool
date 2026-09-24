"""Train the section-3 clinical cohort on its prepared taxonomy and protein targets."""

import argparse
from pathlib import Path

from citepool.representation import RepresentationConfig, learn_representation


DATA = Path(__file__).resolve().parents[2] / "data" / "section3"


def train(output_dir: Path, *, identity_genes: bool = False, linear_ratio: float | None = None) -> None:
    """Read the refined taxonomy and all-protein target used for the final cohort result."""
    if identity_genes and linear_ratio is None:
        raise ValueError("The historical latent-classifier ratio was not recorded; supply --linear-ratio")
    if not (DATA / "rna.h5ad").is_file() or not (DATA / "protein_targets.h5ad").is_file() or not (DATA / "taxonomy/tables").is_dir():
        raise FileNotFoundError(f"Prepared section-3 inputs are missing under {DATA}; see experiments/README.md")
    learn_representation(
        RepresentationConfig(
            rna_h5ad=DATA / "rna.h5ad",
            taxonomy_dir=DATA / "taxonomy",
            protein_target_h5ad=DATA / "protein_targets.h5ad",
            output_dir=output_dir,
            protein_ratio=0.15,
            enable_rna_gene_classifier=identity_genes,
            rna_linear_classifier_ratio=linear_ratio or 0.0,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-genes", action="store_true")
    parser.add_argument("--linear-ratio", type=float)
    options = parser.parse_args()
    train(options.output, identity_genes=options.identity_genes, linear_ratio=options.linear_ratio)
