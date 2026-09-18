"""Command dispatcher for the stable CITEpool baseline."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    commands = (
        "run", "identity", "integrate-tree", "prepare-targets", "train",
        "refine-rna", "benchmark"
    )
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(prog="citepool")
        parser.add_argument("command", choices=commands)
        parser.print_help()
        return
    command = "identity" if sys.argv[1] == "cytofuse" else sys.argv[1]
    if command not in commands:
        raise SystemExit(f"unknown command {command!r}; choose from {commands}")
    sys.argv = [f"citepool {command}", *sys.argv[2:]]
    if command == "run":
        from .workflow import main as command_main
    elif command in {"identity", "cytofuse"}:
        from .identity._engine.api import main as command_main
    elif command == "integrate-tree":
        from .identity._engine.integrated_tree import main as command_main
    elif command == "prepare-targets":
        from .identity._targets import main as command_main
    elif command == "train":
        from .representation._training import main as command_main
    elif command == "refine-rna":
        from .refinement._rna_refinement import main as command_main
    else:
        from .benchmark._metrics import main as command_main
    command_main()


if __name__ == "__main__":
    main()
