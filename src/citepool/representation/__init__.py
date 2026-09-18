"""Identity-constrained RNA representation learning."""
from ..config import TrainingConfig as RepresentationConfig


def learn_representation(config: RepresentationConfig) -> dict[str, object]:
    """Fit SCVI/SCANVI with identity constraints and the protein decoder."""
    from ._training import train_parent_set_scanvi
    return train_parent_set_scanvi(config)


__all__ = ['RepresentationConfig', 'learn_representation']
