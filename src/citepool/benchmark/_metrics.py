"""Stable metrics for parent-set protein-decoder experiments."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from sklearn.metrics import roc_auc_score

from .. import __version__
from ..config import BenchmarkConfig
from .._utils.preprocessing import dense, prepare_unintegrated_pca


MIN_CELLS = 30
MIN_BATCHES = 2
MIN_DIRECTION_AGREEMENT = 0.8
MIN_RAW_RELATIVE_DIFFERENCE = 0.5


def zscore_features(values: np.ndarray) -> np.ndarray:
    """Globally standardize each protein for protein-space scIB."""

    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    result = (values - mean) / std
    if not np.isfinite(result).all():
        raise ValueError("non-finite protein z-score")
    return result.astype(np.float32)


def raw_protein_values(rna_path: Path, proteins: pd.Index):
    """Read raw antibody counts in a requested feature order."""

    data = ad.read_h5ad(rna_path)
    feature = data.var["feature_types"].astype(str).eq("Antibody Capture").to_numpy()
    names = pd.Index(
        data.var.loc[feature, "feature_name"].astype(str)
        if "feature_name" in data.var
        else data.var_names[feature].astype(str)
    )
    columns = names.get_indexer(proteins)
    if np.any(columns < 0):
        raise KeyError(f"raw proteins missing: {proteins[columns < 0].tolist()}")
    return data, dense(data.X[:, feature][:, columns]).astype(np.float32)


# Backward-compatible name used by the pre-package pipeline script.
raw_marker_values = raw_protein_values


def run_scib(
    obs: pd.DataFrame,
    pre: np.ndarray,
    integrated: np.ndarray,
    name: str,
    n_jobs: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run the fixed scIB metric collection and return its aggregate scores."""

    benchmark_data = ad.AnnData(
        X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs.copy()
    )
    benchmark_data.obs["batch"] = (
        benchmark_data.obs.batch.astype(str).astype("category")
    )
    benchmark_data.obs["subtype"] = (
        benchmark_data.obs.subtype.astype(str).astype("category")
    )
    benchmark_data.obsm["X_pre"] = np.asarray(pre, dtype=np.float32)
    benchmark_data.obsm[name] = np.asarray(integrated, dtype=np.float32)
    benchmarker = Benchmarker(
        benchmark_data,
        batch_key="batch",
        label_key="subtype",
        embedding_obsm_keys=[name],
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        pre_integrated_embedding_obsm_key="X_pre",
        n_jobs=n_jobs,
        progress_bar=True,
    )
    benchmarker.prepare()
    benchmarker.benchmark()
    detail = benchmarker.get_results(min_max_scale=False, clean_names=True)
    return {
        "Batch": float(detail.loc[name, "Batch correction"]),
        "Bio": float(detail.loc[name, "Bio conservation"]),
        "Total": float(detail.loc[name, "Total"]),
    }, detail


def reconstruction_metrics(
    reconstructed: ad.AnnData,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calculate per-protein recovery metrics and their macro means."""

    truth = np.asarray(reconstructed.layers["target"], dtype=np.float64)
    prediction = np.asarray(reconstructed.X, dtype=np.float64)
    mask = np.asarray(reconstructed.layers["normalized_mask"], dtype=bool)
    rows: list[dict[str, object]] = []
    for protein_index, protein in enumerate(reconstructed.var_names.astype(str)):
        use = (
            mask[:, protein_index]
            & np.isfinite(truth[:, protein_index])
            & np.isfinite(prediction[:, protein_index])
        )
        observed = truth[use, protein_index]
        estimated = prediction[use, protein_index]
        rmse = float(np.sqrt(np.mean(np.square(estimated - observed))))
        std = float(np.std(observed))
        rows.append({
            "marker": protein,
            "n_values": int(use.sum()),
            "pearson": float(pearsonr(observed, estimated).statistic),
            "spearman": float(spearmanr(observed, estimated).statistic),
            "rmse": rmse,
            "truth_std": std,
            "stdRMSE": rmse / std,
        })
    detail = pd.DataFrame(rows)
    return {
        "蛋白 Pearson": float(detail.pearson.mean()),
        "蛋白 Spearman": float(detail.spearman.mean()),
        "蛋白 stdRMSE": float(detail.stdRMSE.mean()),
    }, detail


def direction_accuracy(
    raw: np.ndarray,
    prediction: np.ndarray,
    obs: pd.DataFrame,
    markers: pd.Index,
    experiment: str,
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score subtype-vs-rest marker direction within batch.

    Candidate marker-subtype pairs are selected only when raw counts show a
    reproducible direction and sufficient relative separation across batches.
    """

    batches = obs.batch.astype(str).to_numpy()
    subtypes = obs.subtype.astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for batch in pd.unique(batches):
        batch_use = batches == batch
        for subtype in sorted(pd.unique(subtypes[batch_use])):
            target = batch_use & (subtypes == subtype)
            reference = batch_use & (subtypes != subtype)
            if target.sum() < MIN_CELLS or reference.sum() < MIN_CELLS:
                continue
            use = target | reference
            y = target[use]
            for marker_index, marker in enumerate(markers):
                raw_values = raw[use, marker_index]
                predicted_values = prediction[use, marker_index]
                raw_iqr = np.quantile(raw_values, 0.75) - np.quantile(raw_values, 0.25)
                predicted_iqr = (
                    np.quantile(predicted_values, 0.75)
                    - np.quantile(predicted_values, 0.25)
                )
                raw_difference = np.median(raw_values[y]) - np.median(raw_values[~y])
                predicted_difference = (
                    np.median(predicted_values[y])
                    - np.median(predicted_values[~y])
                )
                rows.append({
                    "experiment": experiment,
                    "batch": batch,
                    "subtype": subtype,
                    "marker": marker,
                    "n_target": int(target.sum()),
                    "n_reference": int(reference.sum()),
                    "raw_auc": float(roc_auc_score(y, raw_values)),
                    "prediction_auc": float(roc_auc_score(y, predicted_values)),
                    "raw_relative_difference": (
                        float(raw_difference / raw_iqr) if raw_iqr > 1e-12 else 0.0
                    ),
                    "prediction_relative_difference": (
                        float(predicted_difference / predicted_iqr)
                        if predicted_iqr > 1e-12 else 0.0
                    ),
                })
    detail = pd.DataFrame(rows)
    selected_rows: list[dict[str, object]] = []
    evaluated_rows: list[dict[str, object]] = []
    for (subtype, marker), group in detail.groupby(
        ["subtype", "marker"], sort=False
    ):
        fraction_high = float(group.raw_relative_difference.ge(0).mean())
        agreement = max(fraction_high, 1 - fraction_high)
        direction = 1.0 if fraction_high >= 0.5 else -1.0
        oriented_raw = direction * group.raw_relative_difference
        selected = (
            group.batch.nunique() >= MIN_BATCHES
            and agreement >= MIN_DIRECTION_AGREEMENT
            and float(oriented_raw.median()) >= MIN_RAW_RELATIVE_DIFFERENCE
        )
        selected_rows.append({
            "experiment": experiment,
            "subtype": subtype,
            "marker": marker,
            "n_batches": int(group.batch.nunique()),
            "consensus_direction": "high" if direction > 0 else "low",
            "direction_agreement": agreement,
            "median_oriented_raw_relative_difference": float(oriented_raw.median()),
            "selected": selected,
        })
        if selected:
            for record in group.to_dict(orient="records"):
                record["consensus_direction"] = "high" if direction > 0 else "low"
                record["oriented_prediction_relative_difference"] = (
                    direction * record["prediction_relative_difference"]
                )
                record["direction_correct"] = (
                    record["oriented_prediction_relative_difference"] > 0
                )
                evaluated_rows.append(record)
    selection = pd.DataFrame(selected_rows)
    evaluated = pd.DataFrame(evaluated_rows)
    if evaluated.empty:
        raise ValueError(f"{experiment}: no marker-subtype direction pairs selected")
    pairs = (
        evaluated.groupby(["subtype", "marker"], as_index=False)
        .agg(
            n_batches=("batch", "nunique"),
            direction_accuracy=("direction_correct", "mean"),
        )
    )
    return (
        float(pairs.direction_accuracy.mean()),
        detail,
        selection,
        evaluated,
        pairs,
    )


def benchmark_three_experiments(config: BenchmarkConfig) -> pd.DataFrame:
    """Calculate and persist the fixed ten requested metrics."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    for experiment in config.experiments:
        print(f"\n## {experiment}: loading", flush=True)
        run = config.root / experiment
        model = ad.read_h5ad(run / "model/official_scanvi_parent_set.h5ad")
        reconstructed = ad.read_h5ad(run / "model/reconstructed_protein.h5ad")
        proteins = pd.Index(reconstructed.var_names.astype(str))
        rna_path = config.data_root / f"{experiment}.h5ad"
        raw_data, raw = raw_protein_values(rna_path, proteins)
        if not model.obs_names.equals(raw_data.obs_names.astype(str)):
            raise ValueError(f"{experiment}: model/raw cell order differs")
        obs = raw_data.obs[["batch", "subtype"]].copy()

        print(f"## {experiment}: RNA scIB", flush=True)
        pre_rna = prepare_unintegrated_pca(
            rna_path, run / "model/highly_variable_genes.csv", model.obs_names
        )
        rna_scib, rna_detail = run_scib(
            obs, pre_rna, np.asarray(model.obsm["X_scanvi"]),
            "RNA_model", config.n_jobs,
        )
        rna_detail.to_csv(config.output_dir / f"{experiment}_rna_scib_detail.csv")
        del pre_rna
        gc.collect()

        print(f"## {experiment}: protein metrics and scIB", flush=True)
        recovery, recovery_detail = reconstruction_metrics(reconstructed)
        recovery_detail.insert(0, "experiment", experiment)
        recovery_detail.to_csv(
            config.output_dir / f"{experiment}_protein_recovery_by_marker.csv",
            index=False,
        )
        pre_protein = zscore_features(np.log1p(raw))
        reconstructed_values = np.asarray(reconstructed.X, dtype=np.float32)
        protein_scib, protein_detail = run_scib(
            obs,
            pre_protein,
            zscore_features(reconstructed_values),
            "protein_model",
            config.n_jobs,
        )
        protein_detail.to_csv(
            config.output_dir / f"{experiment}_protein_scib_detail.csv"
        )

        print(f"## {experiment}: direction accuracy", flush=True)
        direction_markers = proteins
        direction_raw = raw
        direction_prediction = reconstructed_values
        if config.direction_marker_root is not None:
            marker_reference = ad.read_h5ad(
                config.direction_marker_root / experiment
                / "prepared/cluster_translated_marker_targets.h5ad",
                backed="r",
            )
            direction_markers = pd.Index(marker_reference.var_names.astype(str))
            columns = proteins.get_indexer(direction_markers)
            if np.any(columns < 0):
                raise KeyError(
                    f"{experiment}: direction markers absent from reconstruction: "
                    f"{direction_markers[columns < 0].tolist()}"
                )
            direction_raw = raw[:, columns]
            direction_prediction = reconstructed_values[:, columns]
        direction, raw_direction, selection, evaluated, pairs = direction_accuracy(
            direction_raw,
            direction_prediction,
            obs,
            direction_markers,
            experiment,
        )
        raw_direction.to_csv(
            config.output_dir / f"{experiment}_direction_all_candidates.csv.gz",
            index=False,
        )
        selection.to_csv(
            config.output_dir / f"{experiment}_direction_selection.csv", index=False
        )
        evaluated.to_csv(
            config.output_dir / f"{experiment}_direction_evaluated.csv.gz", index=False
        )
        pairs.to_csv(
            config.output_dir / f"{experiment}_direction_by_marker_subtype.csv",
            index=False,
        )
        runtime = json.loads((run / "model/run_summary.json").read_text())
        summary_rows.append({
            "experiment": experiment,
            "CytoFuse resolution": 1.0,
            "n_cells": int(model.n_obs),
            "n_proteins": int(len(proteins)),
            "n_direction_markers": int(len(direction_markers)),
            "n_parent_cells": int(runtime["n_partial_parent_cells"]),
            "protein_decoder_uses_batch": bool(runtime["protein_decoder_uses_batch"]),
            "scIB Batch": rna_scib["Batch"],
            "scIB Bio": rna_scib["Bio"],
            "scIB Total": rna_scib["Total"],
            **recovery,
            "蛋白scIB batch": protein_scib["Batch"],
            "蛋白scIB bio": protein_scib["Bio"],
            "蛋白scIB total": protein_scib["Total"],
            "蛋白方向正确率": direction,
            "n_direction_marker_subtype_pairs": int(len(pairs)),
            "runtime_min": float(runtime["runtime_minutes"]),
        })
        pd.DataFrame(summary_rows).to_csv(
            config.output_dir / "metrics_partial.csv", index=False
        )
        del model, reconstructed, raw_data, raw, reconstructed_values
        gc.collect()

    summary = pd.DataFrame(summary_rows)
    requested_columns = [
        "experiment", "scIB Batch", "scIB Bio", "scIB Total",
        "蛋白 Pearson", "蛋白 Spearman", "蛋白 stdRMSE",
        "蛋白scIB batch", "蛋白scIB bio", "蛋白scIB total", "蛋白方向正确率",
    ]
    summary.to_csv(
        config.output_dir / "section1_r1p0_metrics_full.csv", index=False
    )
    summary[requested_columns].to_csv(
        config.output_dir / "section1_r1p0_requested_metrics.csv", index=False
    )
    evaluation_config = {
        "citepool_baseline_version": __version__,
        "cytofuse_resolution": 1.0,
        "classifier": "official SCANVI + descendant-set parent likelihood",
        "parent_cells_are_not_independent_classes": True,
        "protein_decoder_uses_batch": False,
        "protein_target": "all-protein CLR then supervision-node-by-batch translation",
        "protein_scib_embedding": "globally z-scored reconstructed protein matrix",
        "protein_scib_preintegration": "globally z-scored log1p raw protein counts",
        "direction_accuracy": {
            "comparison": "original subtype versus all other subtypes within each batch",
            "min_cells": MIN_CELLS,
            "min_batches": MIN_BATCHES,
            "min_direction_agreement": MIN_DIRECTION_AGREEMENT,
            "min_median_oriented_raw_relative_difference": MIN_RAW_RELATIVE_DIFFERENCE,
            "aggregation": (
                "mean per-batch sign accuracy within marker-subtype, "
                "then mean across pairs"
            ),
            "marker_source_root": (
                str(config.direction_marker_root)
                if config.direction_marker_root is not None
                else "all reconstructed proteins"
            ),
        },
    }
    (config.output_dir / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direction-marker-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/section1"))
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = benchmark_three_experiments(BenchmarkConfig(
        root=args.root,
        output_dir=args.output_dir,
        direction_marker_root=args.direction_marker_root,
        data_root=args.data_root,
        n_jobs=args.n_jobs,
    ))
    requested_columns = [
        "experiment", "scIB Batch", "scIB Bio", "scIB Total",
        "蛋白 Pearson", "蛋白 Spearman", "蛋白 stdRMSE",
        "蛋白scIB batch", "蛋白scIB bio", "蛋白scIB total", "蛋白方向正确率",
    ]
    print("\nFinal metrics:\n" + summary[requested_columns].to_string(index=False))


if __name__ == "__main__":
    main()
