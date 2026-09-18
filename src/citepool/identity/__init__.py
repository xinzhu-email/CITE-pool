"""Cell identity from discrete protein-marker configurations.

Protein clustering, marker-state inference, batch alignment, and identity
constraints belong to this module. The bundled engine is an implementation
 detail of CITEpool, not a separately installed algorithm.
"""
from ..config import CytoFuseConfig as ProteinIdentityConfig
from ..config import TargetPreparationConfig


def infer_identity(config: ProteinIdentityConfig) -> dict[str, object]:
    """Infer protein identities without reading biological ground-truth labels."""
    from ._engine.api import run_cytofuse
    return run_cytofuse(config)


def prepare_protein_targets(config: TargetPreparationConfig) -> dict[str, object]:
    """Prepare identity- and batch-aware targets for the protein decoder."""
    from ._targets import prepare_translated_protein_targets
    return prepare_translated_protein_targets(config)


__all__ = ['ProteinIdentityConfig', 'TargetPreparationConfig',
           'infer_identity', 'prepare_protein_targets']
