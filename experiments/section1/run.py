"""Run the section-1 intersection or full-panel CITEpool experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from citepool import run
from citepool.config import WorkflowConfig


ROOT = Path(__file__).resolve().parents[2]
PANELS = {
    "intersection": (),
    "full_panel": ("--protein-mask-obsm-key", "protein_measurement_mask"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANELS, default="intersection")
    parser.add_argument("--expr", choices=("expr1", "expr2", "expr3"), default="expr1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a supported device string")
    args = parser.parse_args()
    input_h5ad = ROOT / "data" / "section1" / args.panel / f"{args.expr}.h5ad"
    if not input_h5ad.is_file():
        parser.error(f"Missing input data: {input_h5ad}")
    output_dir = args.output or ROOT / "results" / f"section1_{args.panel}_{args.expr}_r{args.resolution:g}"

    run(
        WorkflowConfig(
            input_h5ad=input_h5ad,
            output_dir=output_dir,
            all_proteins=True,
            initial_resolution=args.resolution,
            device=args.device,
            enable_rna_refinement=True,
            cytofuse_extra_args=PANELS[args.panel],
            refinement_overrides={
                "dip_cutoff": 0.00495,
                "separation_cutoff": 0.5,
                "small_fragment_max_cells": 50,
                "two_d_bic_gain_per_cell": 0.1,
                "one_d_bic_gain_per_cell": 0.1,
                "min_split_batches": 3 if args.expr == "expr3" else 1,
            },
        )
    )


if __name__ == "__main__":
    main()
