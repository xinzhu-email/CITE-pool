from __future__ import annotations

import inspect
import unittest

from citepool_baseline.cytofuse import runner
from citepool_baseline.cytofuse.methods import cluster, paramshared


class BundledCytoFuseTests(unittest.TestCase):
    def test_runner_uses_only_package_relative_algorithm_imports(self) -> None:
        source = inspect.getsource(runner)
        self.assertIn("from .tree import", source)
        self.assertIn("from .methods.cluster import", source)
        self.assertIn("from .methods.paramshared import", source)
        self.assertNotIn("from forcitepool", source)
        self.assertNotIn("from cytofuse", source)
        self.assertNotIn("sys.path.insert", source)

    def test_algorithm_modules_resolve_inside_baseline_package(self) -> None:
        self.assertTrue(cluster.__name__.startswith("citepool.identity._engine"))
        self.assertTrue(
            paramshared.__name__.startswith("citepool.identity._engine")
        )

    def test_runner_parser_keeps_citepool_resolution_explicit(self) -> None:
        parser = runner.build_parser()
        args = parser.parse_args(["--input", "input.h5ad"])
        self.assertEqual(args.initial_resolution, 0.5)


if __name__ == "__main__":
    unittest.main()
