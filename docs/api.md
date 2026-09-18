# CITEpool API and implementation layout

Public import: `import citepool`; unified model: `citepool.model.CITEPool`.
All implementation lives under `src/citepool`. The `src/citepool_baseline`
namespace contains only compatibility imports for existing experiments and
serialized objects. CytoFuse is an internal protein identity engine at
`citepool.identity._engine`, with no separate dependency.

```text
src/citepool/
  model/_citepool.py          train, outputs, portable save/load
  model/_data.py              count/feature/mask registration
  identity/_engine/           internal marker-state engine
  identity/_targets.py        translated decoder targets
  identity/_supervision.py    optional information filter
  representation/_training.py SCVI/SCANVI and auxiliary decoder/training plan
  representation/_gene_classifier.py
  refinement/_rna_refinement.py
  _utils/                    shared preprocessing/plotting
  benchmark/                 optional historical benchmark adapter
  api.py                     functional interface and run handle
  workflow.py                three-stage orchestration and final refit
  config.py                  immutable stage/workflow configurations
src/citepool_baseline/
  __init__.py                historical imports map to the same implementation
  __main__.py                historical CLI alias
```

Old flat compatibility scripts, duplicated package trees, build products, and
unused standalone-engine platform/atlas utilities have been backed up outside
the source directory. The active identity module still constructs CITEpool's
resolution-aware atlas and parent-set constraints.

## Model lifecycle

`CITEPool.setup_anndata(adata, *, batch_key="batch", layer=None,
protein_expression_obsm_key=None, protein_names_uns_key=None,
protein_measurement_mask_obsm_key=None, feature_type_key="feature_types")`
validates finite nonnegative count data, unique cells/features/markers, batches,
and measured-protein masks. Registration is stored in `adata.uns` and survives
H5AD serialization. AnnData must be in memory. Normalized/log-transformed
expression must not be supplied in the registered count matrix.

`CITEPool(adata_or_h5ad, *, initial_resolution=1.0, all_proteins=True,
enable_rna_refinement=True, batch_key="batch", seed=2026, protein_seed=0)`
constructs an untrained model. For AnnData, the registered batch key takes
precedence; H5AD path input uses the combined RNA/ADT format.

`train(output_dir, *, device="auto", scvi_epochs=100, scanvi_epochs=50,
training_options=None, refinement_options=None)` executes all enabled stages.
The output directory must be new or empty. Advanced option mappings are
validated against the immutable configuration dataclasses before execution.
The canonical count input is retained at `input.h5ad`; model settings are
recorded in `citepool_model.json`, `workflow_config.json`, and stage manifests.
A model instance may train once; use a new instance for another configuration.

`predict(soft=False)` returns identity labels indexed by cell. Soft output
renormalizes the saved parent-constrained classifier probabilities within the
allowed identity set. `get_latent_representation()` returns a NumPy matrix;
`get_reconstructed_protein()` and `get_gene_weights()` return DataFrames.
Protein values use the translated CLR target scale. These getters read fitted
artifacts; they do not run inference on new cells.

`save(directory)` copies the entire trained run and rewrites run paths in JSON
manifests. `CITEPool.load(directory)` validates and opens a saved model or a
legacy workflow. Load does not resume optimization or transfer query cells.

## Independent stage contracts

```python
from pathlib import Path
from citepool import identity, representation, refinement

identity.infer_identity(identity.ProteinIdentityConfig(
    input_h5ad=Path("counts.h5ad"), output_dir=Path("identity")))
targets = identity.prepare_protein_targets(identity.TargetPreparationConfig(
    rna_h5ad=Path("counts.h5ad"), cytofuse_run=Path("identity"),
    marker_reference_h5ad=None, output_dir=Path("prepared")))
representation.learn_representation(representation.RepresentationConfig(
    rna_h5ad=Path("counts.h5ad"), taxonomy_dir=Path("prepared/taxonomy"),
    protein_target_h5ad=Path(targets["target"]),
    output_dir=Path("initial"), accelerator="cpu"))
```

Target preparation returns `target` (protein target H5AD) and `taxonomy`
(prepared identity directory). The example uses the returned target path.
The identity summary describes the inferred marker states and identity
hierarchy. Protein identity artifacts are under `tables/` and include
`cell_taxonomy_assignments.csv`, `taxonomy_nodes.csv`, and `taxonomy_states.csv`.
The representation requires these tables and their explicit parent-set
relations. The first fitted embedding is
`initial/official_scanvi_parent_set.h5ad`.

```python
refinement.refine_identity(refinement.RefinementConfig(
    rna_h5ad=Path("counts.h5ad"),
    initial_model_h5ad=Path("initial/official_scanvi_parent_set.h5ad"),
    taxonomy_dir=Path("prepared/taxonomy"), output_dir=Path("refined_identity")))
```

RNA refinement returns the new taxonomy directory and its diagnostics. To
obtain the final fitted representation, run stage two again on that revised
identity universe and the same translated protein targets from stage one.
`CITEPool.train()` performs this sequence automatically and implements the
existing stage-two supervision fixes. The low-level functions return JSON-
compatible summaries; their paths are explicit and reusable across stages.

The manuscript's three subsections describe these modules. The second
representation fit uses refined identities; it is the same learning module,
not a fourth algorithm component.
