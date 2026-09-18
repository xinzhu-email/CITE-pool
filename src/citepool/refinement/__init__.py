"""Refining cell identity through conserved RNA structure."""
from ..config import RNARefinementConfig as RefinementConfig


def refine_identity(config: RefinementConfig) -> dict[str, object]:
    """Refine identities using recurring RNA structure across batches.

    This exports updated identity constraints. The full model API then refits
    the representation with those constraints; this function does not refit it.
    """
    from ._rna_refinement import refine_taxonomy_with_rna
    return refine_taxonomy_with_rna(config)


__all__ = ['RefinementConfig', 'refine_identity']
