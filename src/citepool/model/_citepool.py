"""Unified model lifecycle for the three-stage CITEpool algorithm."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Mapping

from .. import __version__
from ..api import CITEpoolRun, fit, load_run

if TYPE_CHECKING:
    from anndata import AnnData


class CITEPool:
    """Infer cell identities and learn their constrained RNA representation.

    Inputs may be a registered AnnData or a combined RNA/ADT count H5AD.
    Training performs protein identity inference, constrained SCVI/SCANVI,
    RNA identity refinement, and a final refit. All stages use the package
    defaults unless explicitly overridden. Ground-truth labels are optional
    evaluation annotations and are never identity-inference inputs.

    ``predict`` and the getters return fitted cells. Query transfer and
    resuming optimization are not currently part of this model API.
    """

    def __init__(self, adata: AnnData | str | Path, *,
                 initial_resolution: float = 1.0, all_proteins: bool = True,
                 enable_rna_refinement: bool = True, batch_key: str = 'batch',
                 seed: int = 2026, protein_seed: int = 0):
        from ._data import REGISTRY_KEY
        if isinstance(adata, (str, Path)):
            self.adata = None
            self.input_h5ad = Path(adata).expanduser().resolve()
            if not self.input_h5ad.is_file():
                raise FileNotFoundError(self.input_h5ad)
        else:
            if REGISTRY_KEY not in adata.uns:
                raise ValueError('Call CITEPool.setup_anndata(adata) first')
            self.adata = adata
            self.input_h5ad = None
            batch_key = str(adata.uns[REGISTRY_KEY]['batch_key'])
        if initial_resolution <= 0:
            raise ValueError('initial_resolution must be positive')
        self.init_params_ = dict(initial_resolution=float(initial_resolution),
            all_proteins=all_proteins, enable_rna_refinement=enable_rna_refinement,
            batch_key=batch_key, seed=seed, protein_seed=protein_seed)
        self.run_: CITEpoolRun | None = None

    @classmethod
    def setup_anndata(cls, adata: AnnData, *, batch_key: str = 'batch',
                      layer: str | None = None,
                      protein_expression_obsm_key: str | None = None,
                      protein_names_uns_key: str | None = None,
                      protein_measurement_mask_obsm_key: str | None = None,
                      feature_type_key: str = 'feature_types') -> None:
        """Register count data, batches, and protein columns in ``adata.uns``.

        Supports combined RNA/ADT features, or RNA in X/layers with protein
        counts in obsm. DataFrame protein columns supply marker names; array
        input requires names in uns. Mosaic panels must supply a measurement
        mask: zero counts alone do not indicate an unmeasured protein.
        """
        from ._data import register_anndata
        register_anndata(adata, batch_key=batch_key, layer=layer,
            protein_expression_obsm_key=protein_expression_obsm_key,
            protein_names_uns_key=protein_names_uns_key,
            protein_measurement_mask_obsm_key=protein_measurement_mask_obsm_key,
            feature_type_key=feature_type_key)

    @property
    def is_trained(self) -> bool:
        """Whether final model artifacts are available."""
        return self.run_ is not None and self.run_.complete

    def _require_run(self) -> CITEpoolRun:
        if self.run_ is None:
            raise RuntimeError('Train the model or call CITEPool.load() first')
        return self.run_.validate()

    def train(self, output_dir: str | Path, *, device: str = 'auto',
              scvi_epochs: int = 100, scanvi_epochs: int = 50,
              training_options: Mapping[str, object] | None = None,
              refinement_options: Mapping[str, object] | None = None) -> CITEpoolRun:
        """Train all enabled stages into a new output directory.

        Advanced options use TrainingConfig/RNARefinementConfig field names.
        The saved configuration includes resolved defaults and all overrides.
        An existing nonempty output directory is rejected.
        """
        if self.run_ is not None:
            raise RuntimeError('Use a new model instance for another training run')
        output_dir = Path(output_dir).expanduser().resolve()
        options = dict(self.init_params_)
        options['cytofuse_seed'] = options.pop('protein_seed')
        if self.adata is None:
            # Use the same validation/conversion contract as in-memory input.
            import anndata as ad
            source = ad.read_h5ad(self.input_h5ad)
            self.setup_anndata(source, batch_key=options['batch_key'],
                              protein_measurement_mask_obsm_key='protein_measurement_mask' if 'protein_measurement_mask' in source.obsm else None)
        else:
            source = self.adata
        from ._data import prepare_input
        canonical = prepare_input(source)
        # Internal stages consume canonical obs.batch, regardless of user key.
        options['batch_key'] = 'batch'
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f'output_dir is not empty: {output_dir}')
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='citepool-input-', dir=output_dir.parent) as temporary:
            input_path = Path(temporary) / 'input.h5ad'
            canonical.write_h5ad(input_path)
            result = fit(input_path, output_dir, device=device,
                scvi_epochs=scvi_epochs, scanvi_epochs=scanvi_epochs,
                training_options=training_options, refinement_options=refinement_options,
                **options)
            # Keep registered input with the model for a portable, auditable run.
            shutil.copy2(input_path, output_dir / 'input.h5ad')
        self.run_ = result
        self._relocate_input_paths(input_path, output_dir / 'input.h5ad')
        self.run_ = load_run(output_dir, validate=True)
        (output_dir / 'citepool_model.json').write_text(json.dumps({
            'version': __version__, 'class': 'CITEPool',
            'init_params': self.init_params_, 'input_contract': 'raw_counts',
        }, indent=2), encoding='utf-8')
        return self.run_

    def _relocate_input_paths(self, old: Path, new: Path) -> None:
        """Replace the temporary input location in saved JSON manifests."""
        for p in self.run_.root.rglob('*.json'):
            text = p.read_text(encoding='utf-8')
            if str(old) in text:
                p.write_text(text.replace(str(old), str(new)), encoding='utf-8')

    def get_latent_representation(self):
        """Return fitted-cell × latent-dimension values in input cell order."""
        data = self._require_run().read_embedding(backed='r')
        try:
            return data.obsm['X_scanvi'].copy()
        finally:
            data.file.close()

    def predict(self, *, soft: bool = False):
        """Return final identities, or constrained cell × identity probabilities."""
        import pandas as pd
        data = self._require_run().read_embedding(backed='r')
        try:
            if soft:
                values = data.obsm['leaf_probability_parent_constrained'].copy()
                mass = values.sum(axis=1, keepdims=True)
                if (mass <= 0).any():
                    raise ValueError('Saved identity probabilities contain an empty row')
                return pd.DataFrame(values / mass,
                    index=data.obs_names, columns=list(data.uns['leaf_names']))
            return data.obs['predicted_leaf'].copy().rename('citepool_identity')
        finally:
            data.file.close()

    def get_reconstructed_protein(self):
        """Return fitted-cell × protein values in the translated CLR target scale.

        These are decoder predictions, not raw ADT counts or calibrated
        molecule concentrations.
        """
        import pandas as pd
        from scipy import sparse
        data = self._require_run().read_reconstructed_protein(backed='r')
        try:
            values = data.X[:]
            values = values.toarray() if sparse.issparse(values) else values
            return pd.DataFrame(values, index=data.obs_names, columns=data.var_names)
        finally:
            data.file.close()

    def get_gene_weights(self):
        """Read direct RNA classifier coefficients for identity-gene interpretation."""
        import pandas as pd
        root = self._require_run().final_model_dir / 'rna_gene_classifier'
        path = root / 'rna_gene_classifier_all_weights.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Gene classifier weights not available: {path}')
        return pd.read_csv(path)

    def save(self, directory: str | Path) -> Path:
        """Copy the complete trained run to a new, portable model directory."""
        run = self._require_run()
        destination = Path(directory).expanduser().resolve()
        if destination == run.root.resolve():
            return destination
        if destination.exists():
            raise FileExistsError(destination)
        if run.root.resolve() in destination.parents:
            raise ValueError('Save directory must be outside the existing run')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='citepool-save-', dir=destination.parent) as tmp:
            staged = Path(tmp) / 'model'
            shutil.copytree(run.root, staged)
            for p in staged.rglob('*.json'):
                text = p.read_text(encoding='utf-8')
                p.write_text(text.replace(str(run.root.resolve()), str(destination)), encoding='utf-8')
            staged.rename(destination)
        return destination

    @classmethod
    def load(cls, directory: str | Path) -> CITEPool:
        """Open a saved model or an existing CITEpool workflow without retraining."""
        run = load_run(Path(directory).expanduser().resolve(), validate=True)
        model = cls.__new__(cls)
        model.adata = None
        model.input_h5ad = run.root / 'input.h5ad'
        metadata = run.root / 'citepool_model.json'
        model.init_params_ = json.loads(metadata.read_text())['init_params'] if metadata.is_file() else {}
        model.run_ = run
        return model

    def __repr__(self) -> str:
        state = 'trained' if self.is_trained else 'untrained'
        return f'CITEPool({state}, initial_resolution={self.init_params_.get("initial_resolution", "from saved workflow")})'
