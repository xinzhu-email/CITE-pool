"""Run CITEpool on the bundled section-1 intersection-panel test data."""

from __future__ import annotations

import argparse
from pathlib import Path

from citepool import run
from citepool.config import WorkflowConfig


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/section1/intersection/expr1.h5ad"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/section1_expr1")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a supported device string")
    args = parser.parse_args()

    run(
        WorkflowConfig(
            input_h5ad=INPUT,
            output_dir=args.output,
            all_proteins=True,
            initial_resolution=args.resolution,
            device=args.device,
            enable_rna_refinement=True,
            refinement_overrides={
                "dip_cutoff": 0.00495,
                "separation_cutoff": 0.5,
                "small_fragment_max_cells": 50,
                "two_d_bic_gain_per_cell": 0.1,
                "one_d_bic_gain_per_cell": 0.1,
                "min_split_batches": 1,
            },
        )
    )


if __name__ == "__main__":
    main()
