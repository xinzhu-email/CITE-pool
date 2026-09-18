from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import citepool_baseline as citepool
from citepool_baseline.api import CITEpoolRun


class PublicApiTests(unittest.TestCase):
    def test_fit_builds_safe_baseline_configuration(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_h5ad = root / "input.h5ad"
            input_h5ad.touch()
            output = root / "run"

            with patch(
                "citepool_baseline.workflow.run_workflow",
                return_value={"final_model_dir": str(output / "citepool_initial")},
            ) as mocked:
                result = citepool.fit(
                    input_h5ad,
                    output,
                    device="cpu",
                    initial_resolution=0.5,
                )

            config = mocked.call_args.args[0]
            self.assertIsInstance(result, CITEpoolRun)
            self.assertEqual(config.initial_resolution, 0.5)
            self.assertEqual(config.training_overrides["accelerator"], "cpu")
            self.assertEqual(config.training_overrides["protein_ratio"], 1.0)
            self.assertEqual(
                config.training_overrides["rna_linear_classifier_ratio"],
                0.0,
            )

    def test_fit_rejects_duplicated_advanced_option(self) -> None:
        with TemporaryDirectory() as temporary:
            input_h5ad = Path(temporary) / "input.h5ad"
            input_h5ad.touch()
            with self.assertRaisesRegex(ValueError, "repeats explicit"):
                citepool.fit(
                    input_h5ad,
                    Path(temporary) / "run",
                    training_options={"protein_ratio": 2.0},
                )

    def test_fit_refuses_nonempty_output_by_default(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_h5ad = root / "input.h5ad"
            input_h5ad.touch()
            output = root / "run"
            output.mkdir()
            (output / "old-result.txt").touch()
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                citepool.fit(input_h5ad, output)

    def test_load_run_resolves_artifacts_and_reads_tables(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "citepool_initial"
            model.mkdir()
            (model / "official_scanvi_parent_set.h5ad").touch()
            (model / "final_leaf_assignments.csv").write_text(
                "cell,label\nc1,L1\n",
                encoding="utf-8",
            )
            (model / "metrics.csv").write_text(
                "metric,value\nARI,0.8\n",
                encoding="utf-8",
            )
            (root / "workflow_summary.json").write_text(
                json.dumps({"final_model_dir": "citepool_initial"}),
                encoding="utf-8",
            )

            result = citepool.load_run(root, validate=True)

            self.assertTrue(result.complete)
            self.assertEqual(result.final_model_dir, model)
            self.assertEqual(result.read_metrics().iloc[0]["metric"], "ARI")
            self.assertEqual(result.read_assignments().iloc[0]["label"], "L1")

    def test_moved_run_falls_back_from_stale_absolute_path(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "citepool_initial"
            model.mkdir()
            (root / "workflow_summary.json").write_text(
                json.dumps({"final_model_dir": "/old/location/citepool_initial"}),
                encoding="utf-8",
            )

            result = citepool.load_run(root)

            self.assertEqual(result.final_model_dir, model)


if __name__ == "__main__":
    unittest.main()
