"""Small, typed public API for the CITEpool baseline.

The implementation modules remain independently callable, but users should
normally enter through :func:`fit`, :func:`run`, or :func:`load_run`.  Heavy
scientific dependencies are imported lazily so importing this module is cheap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import RNARefinementConfig, TrainingConfig, WorkflowConfig


PathLike = str | Path


def _jsonable(value: object) -> Any:
    """Convert dataclasses/path values to standard JSON-compatible objects."""

    return json.loads(json.dumps(value, default=str))


def _path(value: PathLike, *, name: str, must_exist: bool = False) -> Path:
    result = Path(value).expanduser()
    if must_exist and not result.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {result}")
    return result


def _summary_path(root: Path, value: object, fallback: Path) -> Path:
    if value is None or str(value).strip() == "":
        return fallback
    result = Path(str(value)).expanduser()
    if not result.is_absolute():
        return root / result
    # A copied run must never read the source's artifacts when both exist.
    local = root / result.name
    if local.exists():
        return local
    if fallback.name == result.name and fallback.exists():
        return fallback
    # Old runs record absolute paths.  Prefer the conventional in-run path
    # when a result directory has subsequently been moved.
    if result.exists() or not fallback.exists():
        return result
    return fallback


def _require(path: Path, *, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


@dataclass(frozen=True)
class CITEpoolRun:
    """Typed handle to one completed or partially completed workflow run."""

    root: Path
    summary: Mapping[str, Any]
    config: Mapping[str, Any] | None = None

    @property
    def cytofuse_dir(self) -> Path:
        return self.root / "cytofuse"

    @property
    def prepared_dir(self) -> Path:
        return self.root / "prepared"

    @property
    def initial_model_dir(self) -> Path:
        return self.root / "citepool_initial"

    @property
    def final_model_dir(self) -> Path:
        return _summary_path(
            self.root,
            self.summary.get("final_model_dir"),
            self.initial_model_dir,
        )

    @property
    def final_taxonomy_dir(self) -> Path:
        return _summary_path(
            self.root,
            self.summary.get("final_taxonomy_dir"),
            self.prepared_dir / "taxonomy",
        )

    @property
    def embedding_h5ad(self) -> Path:
        return self.final_model_dir / "official_scanvi_parent_set.h5ad"

    @property
    def reconstructed_protein_h5ad(self) -> Path:
        return self.final_model_dir / "reconstructed_protein.h5ad"

    @property
    def assignments_csv(self) -> Path:
        return self.final_model_dir / "final_leaf_assignments.csv"

    @property
    def metrics_csv(self) -> Path:
        return self.final_model_dir / "metrics.csv"

    @property
    def complete(self) -> bool:
        """Whether the minimum final-model artifacts are present."""

        required = (self.embedding_h5ad, self.assignments_csv, self.metrics_csv)
        return all(path.is_file() for path in required)

    def validate(self, *, require_protein: bool = False) -> "CITEpoolRun":
        """Validate expected artifacts and return this handle for chaining."""

        required = [self.embedding_h5ad, self.assignments_csv, self.metrics_csv]
        if require_protein:
            required.append(self.reconstructed_protein_h5ad)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"CITEpool run is incomplete; missing: {missing}")
        return self

    def read_embedding(self, *, backed: str | None = None):
        """Read the final SCANVI AnnData object lazily."""

        import anndata as ad

        path = _require(self.embedding_h5ad, description="embedding")
        return ad.read_h5ad(path, backed=backed)

    def read_reconstructed_protein(self, *, backed: str | None = None):
        """Read the reconstructed-protein AnnData object lazily."""

        import anndata as ad

        path = _require(
            self.reconstructed_protein_h5ad,
            description="reconstructed protein",
        )
        return ad.read_h5ad(path, backed=backed)

    def read_assignments(self):
        """Read final flat leaf assignments as a pandas DataFrame."""

        import pandas as pd

        path = _require(self.assignments_csv, description="assignments")
        return pd.read_csv(path)

    def read_metrics(self):
        """Read model metrics as a pandas DataFrame."""

        import pandas as pd

        path = _require(self.metrics_csv, description="metrics")
        return pd.read_csv(path)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the run and key artifacts."""

        return {
            "root": str(self.root),
            "complete": self.complete,
            "final_model_dir": str(self.final_model_dir),
            "final_taxonomy_dir": str(self.final_taxonomy_dir),
            "embedding_h5ad": str(self.embedding_h5ad),
            "reconstructed_protein_h5ad": str(
                self.reconstructed_protein_h5ad
            ),
            "assignments_csv": str(self.assignments_csv),
            "metrics_csv": str(self.metrics_csv),
            "summary": _jsonable(dict(self.summary)),
            "config": (
                _jsonable(dict(self.config))
                if self.config is not None
                else None
            ),
        }


def load_run(output_dir: PathLike, *, validate: bool = False) -> CITEpoolRun:
    """Open an existing workflow directory without loading large matrices."""

    root = Path(output_dir).expanduser()
    summary_path = root / "workflow_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"workflow summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path = root / "workflow_config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else None
    )
    result = CITEpoolRun(root=root, summary=summary, config=config)
    return result.validate() if validate else result


def run(
    config: WorkflowConfig,
    *,
    allow_existing_output: bool = False,
) -> CITEpoolRun:
    """Execute a fully specified workflow and return a typed result handle.

    Parameters
    ----------
    config
        Complete workflow configuration.
    allow_existing_output
        Permit reuse of a non-empty output directory. This is ``False`` by
        default because mixing artifacts from different configurations is a
        common source of apparently inconsistent results.
    """

    normalized = replace(
        config,
        input_h5ad=Path(config.input_h5ad).expanduser(),
        output_dir=Path(config.output_dir).expanduser(),
        marker_reference_h5ad=(
            Path(config.marker_reference_h5ad).expanduser()
            if config.marker_reference_h5ad is not None
            else None
        ),
    )
    _validate_workflow_config(
        normalized,
        allow_existing_output=allow_existing_output,
    )
    from .workflow import run_workflow

    summary = run_workflow(normalized)
    return CITEpoolRun(
        root=normalized.output_dir,
        summary=summary,
        config=asdict(normalized),
    )


def fit(
    input_h5ad: PathLike,
    output_dir: PathLike,
    *,
    initial_resolution: float = 1.0,
    marker_reference_h5ad: PathLike | None = None,
    all_proteins: bool = True,
    batch_key: str = "batch",
    device: str = "auto",
    accelerator: str | None = None,
    devices: int | str = 1,
    seed: int = 2026,
    cytofuse_seed: int = 0,
    scvi_epochs: int = 100,
    scanvi_epochs: int = 50,
    protein_ratio: float = 1.0,
    rna_linear_classifier_ratio: float = 0.0,
    enable_rna_gene_classifier: bool = True,
    enable_rna_refinement: bool = True,
    refinement_parent_leaves: Sequence[str] | None = None,
    unlabel_low_information_clusters: bool = False,
    low_information_max_positive_markers: int = 2,
    cytofuse_args: Sequence[str] = (),
    training_options: Mapping[str, object] | None = None,
    refinement_options: Mapping[str, object] | None = None,
    allow_existing_output: bool = False,
) -> CITEpoolRun:
    """Run CITEpool using the commonly changed options.

    Advanced training and RNA-refinement settings can be supplied through
    ``training_options`` and ``refinement_options``. Explicit keyword
    arguments owned by this function cannot be repeated in
    ``training_options``.
    """

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if accelerator is None:
        accelerator = {"auto": "auto", "cpu": "cpu", "cuda": "gpu"}[device]
    owned_training = {
        "scvi_epochs": int(scvi_epochs),
        "scanvi_epochs": int(scanvi_epochs),
        "protein_ratio": float(protein_ratio),
        "enable_rna_gene_classifier": bool(enable_rna_gene_classifier),
        "rna_linear_classifier_ratio": float(rna_linear_classifier_ratio),
        "accelerator": accelerator,
        "devices": devices,
    }
    advanced_training = dict(training_options or {})
    overlap = sorted(set(owned_training) & set(advanced_training))
    if overlap:
        raise ValueError(
            "training_options repeats explicit fit arguments: "
            f"{overlap}"
        )
    owned_training.update(advanced_training)
    config = WorkflowConfig(
        input_h5ad=_path(input_h5ad, name="input_h5ad", must_exist=True),
        output_dir=Path(output_dir).expanduser(),
        marker_reference_h5ad=(
            _path(
                marker_reference_h5ad,
                name="marker_reference_h5ad",
                must_exist=True,
            )
            if marker_reference_h5ad is not None
            else None
        ),
        all_proteins=bool(all_proteins),
        batch_key=str(batch_key),
        initial_resolution=float(initial_resolution),
        cytofuse_seed=int(cytofuse_seed),
        seed=int(seed),
        device=device,
        enable_rna_refinement=bool(enable_rna_refinement),
        refinement_parent_leaves=(
            tuple(map(str, refinement_parent_leaves))
            if refinement_parent_leaves is not None
            else None
        ),
        cytofuse_extra_args=tuple(map(str, cytofuse_args)),
        unlabel_low_information_clusters=bool(
            unlabel_low_information_clusters
        ),
        low_information_max_positive_markers=int(
            low_information_max_positive_markers
        ),
        training_overrides=owned_training,
        refinement_overrides=dict(refinement_options or {}),
    )
    return run(config, allow_existing_output=allow_existing_output)


def _validate_workflow_config(
    config: WorkflowConfig,
    *,
    allow_existing_output: bool,
) -> None:
    input_path = Path(config.input_h5ad)
    if not input_path.is_file():
        raise FileNotFoundError(f"input_h5ad does not exist: {input_path}")
    if config.marker_reference_h5ad is not None and not Path(
        config.marker_reference_h5ad
    ).is_file():
        raise FileNotFoundError(
            "marker_reference_h5ad does not exist: "
            f"{config.marker_reference_h5ad}"
        )
    if not config.batch_key:
        raise ValueError("batch_key must be non-empty")
    if not config.initial_resolution > 0:
        raise ValueError("initial_resolution must be positive")
    if config.low_information_max_positive_markers < 0:
        raise ValueError(
            "low_information_max_positive_markers must be non-negative"
        )
    training_fields = {item.name for item in fields(TrainingConfig)}
    training_owned = {
        "rna_h5ad",
        "taxonomy_dir",
        "output_dir",
        "protein_target_h5ad",
        "seed",
    }
    invalid_training = set(config.training_overrides) - training_fields
    forbidden_training = set(config.training_overrides) & training_owned
    if invalid_training or forbidden_training:
        raise ValueError(
            "invalid training_options; "
            f"unknown={sorted(invalid_training)}, "
            f"workflow-owned={sorted(forbidden_training)}"
        )
    if config.refinement_overrides and not config.enable_rna_refinement:
        raise ValueError(
            "refinement_options require enable_rna_refinement=True"
        )
    if config.refinement_parent_leaves and not config.enable_rna_refinement:
        raise ValueError(
            "refinement_parent_leaves require enable_rna_refinement=True"
        )
    refinement_fields = {item.name for item in fields(RNARefinementConfig)}
    refinement_owned = {
        "rna_h5ad",
        "initial_model_h5ad",
        "taxonomy_dir",
        "output_dir",
        "batch_key",
        "parent_leaves",
        "seed",
    }
    invalid_refinement = set(config.refinement_overrides) - refinement_fields
    forbidden_refinement = set(config.refinement_overrides) & refinement_owned
    if invalid_refinement or forbidden_refinement:
        raise ValueError(
            "invalid refinement_options; "
            f"unknown={sorted(invalid_refinement)}, "
            f"workflow-owned={sorted(forbidden_refinement)}"
        )
    protein_ratio = config.training_overrides.get("protein_ratio", 1.0)
    if float(protein_ratio) < 0:
        raise ValueError("protein_ratio must be non-negative")
    latent_classifier_ratio = config.training_overrides.get(
        "rna_linear_classifier_ratio",
        0.0,
    )
    if float(latent_classifier_ratio) < 0:
        raise ValueError("rna_linear_classifier_ratio must be non-negative")
    output = Path(config.output_dir)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {output}")
    if (
        output.is_dir()
        and any(output.iterdir())
        and not allow_existing_output
    ):
        raise FileExistsError(
            f"output_dir is not empty: {output}; pass "
            "allow_existing_output=True only when intentional"
        )


__all__ = ["CITEpoolRun", "fit", "load_run", "run"]
