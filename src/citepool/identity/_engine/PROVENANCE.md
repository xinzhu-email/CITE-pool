# Internal protein identity engine provenance

On 2026-09-18 the engine implementation moved to `identity/_engine/`.
The `cytofuse/` directory now contains compatibility imports only. Historical
source mappings below describe the original copy, not a separate dependency.

The CITE-seq CytoFuse engine in this package was copied on 2026-09-08 from the
working tree at `/home/xinzhu/CITEpool/CytoFuse`:

- `forcitepool/run.py` → `cytofuse/runner.py`
- `forcitepool/tree.py` → `cytofuse/tree.py`
- `cytofuse/methods/*.py` → `cytofuse/methods/*.py`

The source repository HEAD at copy time was
`64b71f86d133fc0f2afa7f248dda3e42703e7bc3`, but the source working tree also
contained uncommitted updates. Therefore the commit alone does not identify
the algorithm. Every new CITEpool run records SHA-256 hashes for `runner.py`,
`tree.py`, `methods/cluster.py`, and `methods/paramshared.py` in
`cytofuse/run_summary.json` and the top-level `workflow_summary.json`.

The import paths and source-location fingerprint code were adapted for package
relative execution. No algorithm threshold or fitting rule was intentionally
changed during bundling.

On 2026-09-18 the sole implementation moved into `src/citepool/identity/_engine`.
The original generic `methods/base.py`, `methods/align.py`, and `methods/atlas.py`
were archived: they were not called by CITEpool's workflow. The active
resolution-aware atlas remains in `tree.py`. Historical import compatibility
is now centralized in `src/citepool_baseline/__init__.py`.
