"""Interpretable RNA-to-cell-type linear classifier utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
import torch


class RNAGeneLinearClassifier(torch.nn.Module):
    """Map standardized log-normalized RNA directly to leaf logits.

    The class logits themselves are a class-specific linear RNA encoding.
    There is deliberately no hidden nonlinear layer or batch covariate, so
    every logit has a directly inspectable gene coefficient.
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        gene_mean: np.ndarray,
        gene_std: np.ndarray,
        *,
        target_sum: float = 1e4,
        clip_value: float = 10.0,
    ) -> None:
        super().__init__()
        if gene_mean.shape != (n_genes,) or gene_std.shape != (n_genes,):
            raise ValueError("gene normalization statistics have the wrong shape")
        self.linear = torch.nn.Linear(n_genes, n_classes)
        self.register_buffer(
            "gene_mean", torch.as_tensor(gene_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "gene_std", torch.as_tensor(gene_std, dtype=torch.float32)
        )
        self.target_sum = float(target_sum)
        self.clip_value = float(clip_value)

    def standardized_expression(self, counts: torch.Tensor) -> torch.Tensor:
        if counts.layout != torch.strided:
            counts = counts.to_dense()
        counts = counts.float()
        library_size = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        log_normalized = torch.log1p(counts * (self.target_sum / library_size))
        standardized = (log_normalized - self.gene_mean) / self.gene_std
        return standardized.clamp(-self.clip_value, self.clip_value)

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        return self.linear(self.standardized_expression(counts))


def compute_log_normalized_gene_stats(
    counts: sparse.spmatrix | np.ndarray,
    *,
    target_sum: float = 1e4,
    min_std: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact sparse mean/std for library-normalized log1p RNA."""

    matrix = (
        counts.tocsr().astype(np.float64, copy=True)
        if sparse.issparse(counts)
        else sparse.csr_matrix(np.asarray(counts, dtype=np.float64))
    )
    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(
        float(target_sum), library_size,
        out=np.zeros_like(library_size, dtype=np.float64),
        where=library_size > 0,
    )
    matrix = matrix.multiply(scale[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    squared = matrix.copy()
    squared.data **= 2
    variance = np.asarray(squared.mean(axis=0)).ravel() - mean ** 2
    std = np.sqrt(np.maximum(variance, 0.0))
    std[std < min_std] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def predict_gene_classifier(
    module: RNAGeneLinearClassifier,
    counts: sparse.spmatrix | np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Return softmax probabilities without materializing the full RNA matrix."""

    module.eval()
    device = next(module.parameters()).device
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, counts.shape[0], batch_size):
            stop = min(start + batch_size, counts.shape[0])
            chunk = counts[start:stop]
            if sparse.issparse(chunk):
                chunk = chunk.toarray()
            tensor = torch.as_tensor(
                np.asarray(chunk, dtype=np.float32), device=device
            )
            probabilities.append(
                torch.softmax(module(tensor), dim=-1).cpu().numpy()
            )
    return np.concatenate(probabilities, axis=0).astype(np.float32)


def gene_weight_table(
    module: RNAGeneLinearClassifier,
    gene_names: list[str],
    class_names: list[str],
) -> pd.DataFrame:
    """Export centered multiclass coefficients and within-class ranks."""

    raw = module.linear.weight.detach().cpu().numpy().astype(np.float64)
    # Softmax is invariant to adding the same gene coefficient to every
    # class. Centering across classes removes this non-identifiable component.
    centered = raw - raw.mean(axis=0, keepdims=True)
    log_expression = centered / module.gene_std.detach().cpu().numpy()[None, :]
    rows: list[dict[str, object]] = []
    for class_index, class_name in enumerate(class_names):
        positive_order = np.argsort(-centered[class_index])
        absolute_order = np.argsort(-np.abs(centered[class_index]))
        positive_rank = np.empty(len(gene_names), dtype=int)
        absolute_rank = np.empty(len(gene_names), dtype=int)
        positive_rank[positive_order] = np.arange(1, len(gene_names) + 1)
        absolute_rank[absolute_order] = np.arange(1, len(gene_names) + 1)
        for gene_index, gene in enumerate(gene_names):
            rows.append({
                "cell_type": class_name,
                "gene": gene,
                "raw_standardized_weight": raw[class_index, gene_index],
                "centered_standardized_weight": centered[class_index, gene_index],
                "absolute_centered_weight": abs(centered[class_index, gene_index]),
                "centered_log_expression_weight": log_expression[class_index, gene_index],
                "positive_rank": int(positive_rank[gene_index]),
                "absolute_rank": int(absolute_rank[gene_index]),
            })
    return pd.DataFrame(rows)


def write_gene_weight_reports(
    module: RNAGeneLinearClassifier,
    gene_names: list[str],
    class_names: list[str],
    output_dir: Path,
    *,
    top_n: int = 15,
) -> pd.DataFrame:
    """Write complete coefficients, top genes, and a compact bar plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    weights = gene_weight_table(module, gene_names, class_names)
    weights.to_csv(output_dir / "rna_gene_classifier_all_weights.csv", index=False)
    top = (
        weights.sort_values(["cell_type", "positive_rank"])
        .groupby("cell_type", sort=False, observed=True)
        .head(top_n)
        .copy()
    )
    top.to_csv(output_dir / "rna_gene_classifier_top_positive_genes.csv", index=False)
    negative = (
        weights.sort_values(
            ["cell_type", "centered_standardized_weight"],
            ascending=[True, True],
        )
        .groupby("cell_type", sort=False, observed=True)
        .head(top_n)
        .copy()
    )
    negative.to_csv(output_dir / "rna_gene_classifier_top_negative_genes.csv", index=False)

    n_columns = 3
    n_rows = int(np.ceil(len(class_names) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(15.5, max(5.5, 3.35 * n_rows)),
        squeeze=False,
    )
    colors = plt.get_cmap("tab20").colors
    for class_index, (axis, class_name) in enumerate(zip(axes.flat, class_names)):
        panel = top.loc[top["cell_type"].eq(class_name)].sort_values(
            "centered_standardized_weight", ascending=True
        )
        axis.barh(
            panel["gene"], panel["centered_standardized_weight"],
            color=colors[class_index % len(colors)], alpha=0.86,
        )
        display = class_name.replace("ALIGNED_", "C").replace("__RNA_LEAF_", "-")
        axis.set_title(display, fontsize=10.5, fontweight="bold")
        axis.tick_params(axis="y", labelsize=8)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        axis.set_axisbelow(True)
    for axis in axes.flat[len(class_names):]:
        axis.set_visible(False)
    figure.suptitle(
        "Direct RNA linear classifier: top positive genes per leaf",
        fontsize=17, fontweight="bold", y=0.995,
    )
    figure.supxlabel("Centered coefficient on standardized log-normalized RNA", fontsize=10)
    figure.tight_layout(rect=(0, 0.025, 1, 0.97))
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_dir / f"rna_gene_classifier_top_positive_genes.{suffix}",
            dpi=280, bbox_inches="tight",
        )
    plt.close(figure)
    return weights
