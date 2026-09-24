# Experiment settings

The result numbering is: section 1 = intersection and full mosaic-panel
benchmarks; section 2 = joint reference/query; section 3 = clinical cohort;
section 4 = spatial RNA transfer.

Only `data/section1/intersection/expr1.h5ad` is included in this repository.
Each `run.py` reads data from the paths recorded in its section's JSON file;
the other inputs must be placed at those paths before running. The scripts
raise a clear missing-input error when a required file is absent.

Section 1 uses the current CITEpool API. Its two protein panels share the
same settings; the full panel additionally uses
`protein_measurement_mask`. Run the bundled example with:

```bash
python experiments/section1/run.py
```

For another case, provide `--panel`, `--expr`, `--resolution`, and `--output`,
for example:

```bash
python experiments/section1/run.py --panel full_panel --expr expr3 --resolution 1.5 --output results/full_panel_expr3_r1.5
```

Sections 2–4 record the prepared inputs and the key settings of older
experiments. Their historical data-preparation and algorithm versions are not
part of this repository, so rerunning with the current package does not
guarantee identical results. The section 3 identity-gene branch also requires
the historical latent-classifier ratio, which was not recorded.
