# CITEpool

CITEpool infers cell identity from discrete protein-marker configurations,
learns an identity-constrained RNA representation, and refines identities
through conserved RNA structure. Its protein identity engine (derived from
CytoFuse) is included inside CITEpool; no separate CytoFuse installation or
source checkout is needed.

## Installation

Python 3.10 or newer is required. Install the package from this directory:

```bash
pip install .
```

Or install the distributable wheel:

```bash
pip install /path/to/citepool-3.0.1-py3-none-any.whl
```

For development use `pip install -e .`. Benchmark scoring is optional:
`pip install '.[benchmark]'`. The training integration currently supports
scvi-tools 1.3.x; this is declared in the package dependencies.

## Run the bundled section-1 test data

A fresh clone includes one complete 72-protein, two-batch section-1 input at
`data/section1/intersection/expr1.h5ad`.  It is the only dataset distributed
with this repository.  Install the package, then run:

```bash
python experiments/section1/run.py --output results/section1_expr1
```

The command runs protein identity inference, RNA representation learning and
RNA refinement.  Use `--device cpu` when no CUDA device is available, or
`--resolution 1.5` to change the protein initialization resolution.  The
published experiment settings are in
[`experiments/section1/parameters.json`](experiments/section1/parameters.json).

## Model API

The lifecycle follows the familiar `setup_anndata → model → train → getters`
pattern. Input RNA and ADT must be raw counts. For RNA AnnData with ADT in obsm:

```python
import anndata as ad
from citepool.model import CITEPool

adata = ad.read_h5ad("rna_with_adt.h5ad")
CITEPool.setup_anndata(
    adata,
    batch_key="batch",
    layer="counts",                         # omit if X contains raw counts
    protein_expression_obsm_key="protein_counts",
    protein_names_uns_key="protein_names", # omit for DataFrame protein input
    protein_measurement_mask_obsm_key="protein_measurement_mask",
)
model = CITEPool(adata, initial_resolution=1.0)
model.train("my_run", device="auto")

latent = model.get_latent_representation()   # cells × latent dimensions
identity = model.predict()                  # named pandas Series
probability = model.predict(soft=True)      # cells × identities, row sums = 1
protein = model.get_reconstructed_protein() # cells × proteins, translated CLR scale
weights = model.get_gene_weights()          # direct RNA classifier coefficients

model.save("saved_model")
restored = CITEPool.load("saved_model")
```

`layer=None` selects X. Protein DataFrame columns supply marker names; an
array requires names in uns. A missing protein in a mosaic panel must be
marked unmeasured in the measurement mask, rather than encoded as an observed
zero. The mask has shape cells × proteins; combined-feature input also accepts
cells × all features. Each cell must have at least one measured protein.

For an existing combined RNA/ADT H5AD (such as section1 input), use:

```python
model = CITEPool("combined_counts.h5ad", initial_resolution=1.5)
model.train("my_run", device="cuda")
```

Combined input requires `var["feature_types"]` values `"Gene Expression"` and
`"Antibody Capture"`. To register an in-memory combined object, omit
`protein_expression_obsm_key`. A custom feature-type key can be registered
with `feature_type_key=...`; any batch column can be registered via `batch_key`.
The internal conversion preserves RNA/protein counts and measurement masks,
and does not overwrite the user's expression matrix.

The model getters operate on the fitted cells in input order. `load` also
opens completed legacy workflow directories, without retraining. `save` copies
the complete run, including weights, input counts, identity tables, and
configuration; it rejects an existing destination. Both training and saving
require an explicit output location. The current API does not implement query
transfer or resuming optimization. Protein predictions are in translated CLR
units, rather than raw counts.

## Three algorithm modules

All actual implementations live under `src/citepool`, organized into these three modules:

| Scientific component | Public module | Entry points |
| --- | --- | --- |
| Cell identity from discrete protein-marker configurations | `citepool.identity` | `infer_identity`, `prepare_protein_targets` |
| Identity-constrained RNA representation learning | `citepool.representation` | `learn_representation` |
| Refining cell identity through conserved RNA structure | `citepool.refinement` | `refine_identity` |

Each stage accepts a typed configuration. This lets users run a stage on
existing artifacts without going through the complete model lifecycle:

```python
from pathlib import Path
from citepool.identity import ProteinIdentityConfig, infer_identity

summary = infer_identity(ProteinIdentityConfig(
    input_h5ad=Path("combined_counts.h5ad"),
    output_dir=Path("protein_identity"),
    initial_resolution=1.0,
    device="cpu",
))
```

`RepresentationConfig` describes the raw RNA input, identity-table directory,
protein targets, model settings, and output directory. `RefinementConfig`
describes the raw input, first fitted embedding, identity tables, and RNA split
settings. Stage-three `refine_identity` exports revised constraints; the full
`CITEPool.train()` API subsequently refits the representation with them.
See [docs/api.md](docs/api.md) for exact artifact contracts and configuration examples.

## Defaults and advanced settings

Defaults match the final section1/section2 configuration: all measured
proteins, taxonomy identity readout, initial resolution 1.0, RNA refinement
on, protein seed 0, training seed 2026, and SCVI/SCANVI epochs 100/50.
RNA separation cutoff is 0.5, dip cutoff 0.00495, BIC gain per cell 0.1
for 1D and 2D candidates, fragment threshold 50, minimum child size 50, and
minimum batch size 100. Split support is `max(1, total_batches // 2)` and
remains fixed through recursion (2/3/7 batches require 1/1/3 supporters).
Biological ground-truth labels are optional evaluation annotations; they are
not required for identity inference or training. No additional hard lineage,
null-hypothesis, or differential-expression split guard has been added.

```python
model.train(
    "custom_run", device="cpu", scvi_epochs=100, scanvi_epochs=50,
    training_options={"batch_size": 512, "n_latent": 32},
    refinement_options={"dip_cutoff": 0.00495, "separation_cutoff": 0.5,
                        "two_d_bic_gain_per_cell": 0.1},
)
```

Option names are the fields of `RepresentationConfig` and `RefinementConfig`.
Workflow-owned input/output fields and duplicated explicit arguments are
rejected. To disable stage-three refinement, construct the model with
`enable_rna_refinement=False`.

The functional API remains available: `citepool.fit(input_h5ad, output_dir)`
or `citepool.run(WorkflowConfig(...))`, returning a `CITEpoolRun` artifact
handle. The command line is `citepool run --input ... --output ...` or
`python -m citepool run ...`; `citepool identity ...` runs only stage one.
Legacy `citepool_baseline` imports and command names continue to work through
one compatibility namespace under `src/citepool_baseline`. Historical artifact names are retained to support
existing runs. Experiments and result directories are outside this package.

## Validation

Run the regression and API tests with:

```bash
python -m unittest discover -s tests -v
```

The package includes runnable examples in [examples](examples). Scientific
API contracts are documented in
[docs/api.md](docs/api.md); source provenance is recorded in
[src/citepool/identity/_engine/PROVENANCE.md](src/citepool/identity/_engine/PROVENANCE.md).

## Source layout

```text
citepool_baseline/
  pyproject.toml
  README.md
  src/
    citepool/                 the only algorithm implementation
      model/                  setup, train, getters, save/load
      identity/               discrete protein-marker identities
      representation/         constrained RNA learning
      refinement/             conserved RNA structure
      _utils/                 internal shared helpers
      benchmark/              optional historical benchmark adapter
    citepool_baseline/        two-file legacy namespace; no algorithm copies
  tests/
  examples/
  docs/
```

Build/cache artifacts and old documentation are outside this source directory.
New release archives are written to `CITEpool/releases/citepool/3.0.1/`.
