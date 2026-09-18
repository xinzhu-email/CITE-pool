"""Compatibility namespace for historical experiments and serialized objects.

All implementations live in ``citepool``. A namespace-specific import hook
maps old module paths to those same objects, without duplicate source files.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys

import citepool as _public
from citepool import *  # noqa: F403
from citepool.config import (BenchmarkConfig, CytoFuseConfig, RNARefinementConfig,
                             TargetPreparationConfig, TrainingConfig, WorkflowConfig)

__version__ = _public.__version__

_MODULES = {
    'cytofuse': 'identity._engine',
    'training': 'representation._training',
    'linear_gene_classifier': 'representation._gene_classifier',
    'rna_refinement': 'refinement._rna_refinement',
    'targets': 'identity._targets',
    'supervision_filter': 'identity._supervision',
    'preprocessing': '_utils.preprocessing',
    'plotting': '_utils.plotting',
    'metrics': 'benchmark._metrics',
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def create_module(self, spec):
        return importlib.import_module(self.canonical)

    def exec_module(self, module):
        pass


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith('citepool_baseline.'):
            return None
        suffix = fullname[len('citepool_baseline.'):]
        if suffix == '__main__':
            return None
        first, dot, tail = suffix.partition('.')
        canonical = 'citepool.' + _MODULES.get(first, first) + (dot + tail if dot else '')
        try:
            spec = importlib.util.find_spec(canonical)
        except ModuleNotFoundError:
            return None
        if spec is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(canonical),
            is_package=spec.submodule_search_locations is not None)


if not any(isinstance(finder, _LegacyFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LegacyFinder())


def prepare_translated_protein_targets(config):
    return _public.identity.prepare_protein_targets(config)


def train_parent_set_scanvi(config):
    return _public.representation.learn_representation(config)


def run_cytofuse(config):
    return _public.identity.infer_identity(config)


def build_integrated_hierarchy(cytofuse_run):
    from citepool.identity._engine.integrated_tree import build_integrated_hierarchy as impl
    return impl(cytofuse_run)


def refine_taxonomy_with_rna(config):
    return _public.refinement.refine_identity(config)


def run_workflow(config):
    from citepool.workflow import run_workflow as impl
    return impl(config)


def mark_low_information_clusters(cytofuse_run, *, max_positive_markers=2):
    from citepool.identity._supervision import mark_low_information_clusters as impl
    return impl(cytofuse_run, max_positive_markers=max_positive_markers)


def benchmark_three_experiments(config):
    from citepool.benchmark._metrics import benchmark_three_experiments as impl
    return impl(config)


__all__ = list(_public.__all__) + [
    'BenchmarkConfig', 'CytoFuseConfig', 'RNARefinementConfig', 'TrainingConfig',
    'prepare_translated_protein_targets', 'train_parent_set_scanvi', 'run_cytofuse',
    'build_integrated_hierarchy', 'refine_taxonomy_with_rna', 'run_workflow',
    'mark_low_information_clusters', 'benchmark_three_experiments',
]
