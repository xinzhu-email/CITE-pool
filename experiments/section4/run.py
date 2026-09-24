"""Train the section-4 spatial transfer on its prepared reference/query input."""

import argparse
from pathlib import Path

from citepool.representation import RepresentationConfig, learn_representation


DATA = Path(__file__).resolve().parents[2] / "data" / "section4"


def train(output_dir: Path) -> None:
    """Read the prepared balanced reference/spatial query data and train the old workflow."""
    if not (DATA / "input/rna_refrep_plus_r1clean_query.h5ad").is_file() or not (DATA / "input/protein_refrep_rawclr_r1clean_query_masked.h5ad").is_file() or not (DATA / "taxonomy/tables").is_dir():
        raise FileNotFoundError(f"Prepared section-4 inputs are missing under {DATA}; see experiments/README.md")
    learn_representation(
        RepresentationConfig(
            rna_h5ad=DATA / "input/rna_refrep_plus_r1clean_query.h5ad",
            taxonomy_dir=DATA / "taxonomy",
            protein_target_h5ad=DATA / "input/protein_refrep_rawclr_r1clean_query_masked.h5ad",
            output_dir=output_dir,
            n_hvg=5001,
            gene_min_cells=1,
            scvi_epochs=35,
            scanvi_epochs=25,
            batch_size=512,
            protein_ratio=5.0,
            protein_point_loss="mse",
            protein_hidden_dim=32,
            protein_marker_weights={"CD20": 20.0},
            use_resolution_atlas_parent_sets=False,
            compute_umap=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    train(options.output)
