"""Build one atlas-first hierarchy from current CytoFuse outputs.

CytoFuse intentionally exports marker-similarity taxonomy merges and
resolution-aware coarse-to-fine relations separately.  CITEpool needs one
single-parent graph for descendant-set supervision.  This module treats the
selected coarse-to-fine edges as the fixed skeleton, then replays taxonomy
merges over the resulting atlas components.  A taxonomy merge that is already
contained by one atlas component is collapsed instead of creating a competing
parent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SYNTHETIC_ROOT = "INTEGRATED_ROOT"


def _validate_parent_map(parent_of: dict[str, str]) -> None:
    """Reject cycles in a graph where every child has at most one parent."""

    for start in set(parent_of) | set(parent_of.values()):
        seen: set[str] = set()
        node = str(start)
        while node in parent_of:
            if node in seen:
                raise ValueError(f"cycle in integrated hierarchy at {node}")
            seen.add(node)
            node = parent_of[node]


def _root_of(node: str, parent_of: dict[str, str]) -> str:
    """Return the current top-level representative of a hierarchy node."""

    node = str(node)
    seen: set[str] = set()
    while node in parent_of:
        if node in seen:
            raise ValueError(f"cycle in integrated hierarchy at {node}")
        seen.add(node)
        node = str(parent_of[node])
    return node


def _descendant_leaves(
    node: str,
    children: dict[str, list[str]],
    memo: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if node in memo:
        return memo[node]
    child_nodes = children.get(node, [])
    if not child_nodes:
        answer = (node,)
    else:
        leaves: set[str] = set()
        for child in child_nodes:
            leaves.update(_descendant_leaves(child, children, memo))
        answer = tuple(sorted(leaves))
    memo[node] = answer
    return answer


def build_integrated_hierarchy(cytofuse_run: Path) -> dict[str, object]:
    """Create and persist the atlas-first single-parent CITEpool hierarchy."""

    tables = Path(cytofuse_run) / "tables"
    required = [
        tables / "alignment_membership.csv",
        tables / "taxonomy_merges.csv",
        tables / "cell_taxonomy_assignments.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing CytoFuse hierarchy inputs: {missing}")

    membership = pd.read_csv(tables / "alignment_membership.csv", dtype=str)
    aligned_nodes = sorted(
        membership["aligned_node_id"].dropna().astype(str).unique()
    )
    aligned_set = set(aligned_nodes)
    try:
        merges = pd.read_csv(tables / "taxonomy_merges.csv", dtype=str)
    except pd.errors.EmptyDataError:
        # Older engines wrote a blank CSV when no identities were merged.
        merges = pd.DataFrame(columns=["parent", "child_a", "child_b", "round"])
    if not merges.empty:
        for column in ("parent", "child_a", "child_b"):
            if column not in merges:
                raise KeyError(f"taxonomy_merges.csv lacks {column!r}")
        if "round" in merges:
            merges["_round_numeric"] = pd.to_numeric(
                merges["round"], errors="coerce"
            ).fillna(np.inf)
            merges = merges.sort_values(["_round_numeric", "parent"])
        else:
            merges = merges.sort_values("parent")

    atlas_path = tables / "resolution_atlas_edges.csv"
    atlas_edges = (
        pd.read_csv(atlas_path, dtype=str)
        if atlas_path.exists()
        else pd.DataFrame(columns=["parent", "child", "relation"])
    )
    if not {"parent", "child"}.issubset(atlas_edges.columns):
        raise KeyError("resolution_atlas_edges.csv requires parent and child")
    if "relation" in atlas_edges:
        atlas_edges = atlas_edges.loc[
            atlas_edges["relation"].astype(str).eq("coarse_to_fine")
        ].copy()
    atlas_edges = atlas_edges[["parent", "child"]].drop_duplicates()

    unknown_atlas = (
        set(atlas_edges["parent"].astype(str))
        | set(atlas_edges["child"].astype(str))
    ) - aligned_set
    if unknown_atlas:
        raise ValueError(
            "resolution atlas refers to unknown aligned nodes: "
            f"{sorted(unknown_atlas)}"
        )
    duplicated_children = atlas_edges["child"].astype(str).duplicated(keep=False)
    if duplicated_children.any():
        values = sorted(atlas_edges.loc[duplicated_children, "child"].astype(str).unique())
        raise ValueError(f"atlas child has multiple selected parents: {values}")

    edge_rows: list[dict[str, str]] = []
    parent_of: dict[str, str] = {}
    skipped_atlas_cycles: list[dict[str, str]] = []
    n_atlas_edges_kept = 0
    for row in atlas_edges.itertuples(index=False):
        parent, child = str(row.parent), str(row.child)
        # Ambiguous resolution-atlas calls can contain both A -> B and
        # B -> A.  Keep the first deterministic edge and drop only a later
        # edge that would make the required single-parent graph cyclic.
        cursor = parent
        seen: set[str] = set()
        creates_cycle = parent == child
        while not creates_cycle and cursor in parent_of:
            if cursor in seen:
                creates_cycle = True
                break
            seen.add(cursor)
            cursor = parent_of[cursor]
            creates_cycle = cursor == child
        if creates_cycle:
            skipped_atlas_cycles.append({"parent": parent, "child": child})
            continue
        parent_of[child] = parent
        n_atlas_edges_kept += 1
        edge_rows.append({
            "parent": parent,
            "child": child,
            "relation": "coarse_to_fine",
            "source": "resolution_atlas",
        })
    _validate_parent_map(parent_of)

    # ``taxonomy_rep`` is used only while replaying taxonomy merges.  Aligned
    # nodes themselves remain observable nodes in the final graph; the
    # representative is their atlas-component root for topology construction.
    taxonomy_rep = {
        node: _root_of(node, parent_of) for node in aligned_nodes
    }
    node_mapping = {node: node for node in aligned_nodes}
    mapping_reason = {node: "retained_aligned_node" for node in aligned_nodes}
    audit_rows: list[dict[str, object]] = []
    retained_taxa: set[str] = set()

    for row in merges.itertuples(index=False):
        parent = str(row.parent)
        child_a, child_b = str(row.child_a), str(row.child_b)
        if child_a not in taxonomy_rep or child_b not in taxonomy_rep:
            raise ValueError(
                f"taxonomy merge {parent} references unavailable children "
                f"{child_a}, {child_b}"
            )
        left = _root_of(taxonomy_rep[child_a], parent_of)
        right = _root_of(taxonomy_rep[child_b], parent_of)
        if left == right:
            taxonomy_rep[parent] = left
            node_mapping[parent] = left
            mapping_reason[parent] = "collapsed_into_atlas_component"
            action = "collapsed_existing_component"
            integrated_node = left
        else:
            if left in parent_of or right in parent_of:
                raise RuntimeError(
                    "taxonomy replay attempted to attach a non-root component"
                )
            parent_of[left] = parent
            parent_of[right] = parent
            edge_rows.extend([
                {
                    "parent": parent,
                    "child": left,
                    "relation": "configuration_merge",
                    "source": "taxonomy",
                },
                {
                    "parent": parent,
                    "child": right,
                    "relation": "configuration_merge",
                    "source": "taxonomy",
                },
            ])
            taxonomy_rep[parent] = parent
            node_mapping[parent] = parent
            mapping_reason[parent] = "retained_taxonomy_merge"
            retained_taxa.add(parent)
            action = "created_taxonomy_parent"
            integrated_node = parent
        audit_rows.append({
            "round": getattr(row, "round", ""),
            "taxonomy_parent": parent,
            "taxonomy_child_a": child_a,
            "taxonomy_child_b": child_b,
            "resolved_child_a": left,
            "resolved_child_b": right,
            "action": action,
            "integrated_node": integrated_node,
        })
        _validate_parent_map(parent_of)

    included_nodes = aligned_set | retained_taxa
    roots = sorted(included_nodes - set(parent_of))
    if len(roots) > 1:
        if SYNTHETIC_ROOT in included_nodes:
            raise ValueError(f"reserved node name already exists: {SYNTHETIC_ROOT}")
        for root in roots:
            parent_of[root] = SYNTHETIC_ROOT
            edge_rows.append({
                "parent": SYNTHETIC_ROOT,
                "child": root,
                "relation": "synthetic_root",
                "source": "citepool_baseline",
            })
        included_nodes.add(SYNTHETIC_ROOT)
        roots = [SYNTHETIC_ROOT]
    _validate_parent_map(parent_of)

    edges = pd.DataFrame(
        edge_rows,
        columns=["parent", "child", "relation", "source"],
    ).drop_duplicates()
    if edges["child"].duplicated().any():
        duplicate = sorted(edges.loc[edges["child"].duplicated(False), "child"].unique())
        raise ValueError(f"integrated hierarchy has multiple parents: {duplicate}")

    children = {
        str(parent): sorted(group["child"].astype(str).unique())
        for parent, group in edges.groupby("parent", sort=True)
    }
    leaf_memo: dict[str, tuple[str, ...]] = {}
    node_rows = []
    for node in sorted(included_nodes):
        leaves = _descendant_leaves(node, children, leaf_memo)
        node_rows.append({
            "node_id": node,
            "node_kind": "internal" if node in children else "leaf",
            "is_root": node in roots,
            "source_kind": (
                "synthetic_root" if node == SYNTHETIC_ROOT
                else "taxonomy" if node.startswith("TAXON_")
                else "aligned"
            ),
            "n_descendant_leaves": len(leaves),
            "descendant_leaf_ids": ";".join(leaves),
        })
    nodes = pd.DataFrame(node_rows)
    node_kind = nodes.set_index("node_id")["node_kind"].to_dict()

    cells_path = tables / "cell_taxonomy_assignments.csv"
    cells = pd.read_csv(cells_path, low_memory=False)
    if "cytofuse_final_id" not in cells or "aligned_leaf_id" not in cells:
        raise KeyError("cell assignments require cytofuse_final_id and aligned_leaf_id")
    final_id = cells["cytofuse_final_id"].astype(str)
    supervision = final_id.map(node_mapping).fillna(final_id)
    atlas_parent_nodes = set(atlas_edges["parent"].astype(str))
    atlas_parent_cell = np.zeros(len(cells), dtype=bool)
    if "atlas_node_id" in cells:
        atlas_node = cells["atlas_node_id"].astype(str)
        atlas_parent_cell = atlas_node.isin(atlas_parent_nodes).to_numpy()
        supervision.loc[atlas_parent_cell] = atlas_node.loc[atlas_parent_cell]
    unknown_supervision = set(supervision.astype(str)) - set(nodes["node_id"])
    if unknown_supervision:
        raise ValueError(
            "integrated supervision refers to absent nodes: "
            f"{sorted(unknown_supervision)}"
        )

    def path_for(node: str) -> str:
        path = [str(node)]
        while path[-1] in parent_of:
            path.append(parent_of[path[-1]])
        return ";".join(reversed(path))

    cells["integrated_supervision_node_id"] = supervision.astype(str)
    # The graph retains taxonomy-merge children for audit and descendant
    # traversal, but a node selected by CytoFuse's final cut is itself a
    # terminal label.  Only cells explicitly promoted to a resolution-atlas
    # coarse parent should receive internal/parent-set supervision.
    cells["integrated_graph_node_kind"] = supervision.astype(str).map(node_kind)
    supervision_kind = pd.Series("leaf", index=cells.index, dtype=str)
    supervision_kind.loc[atlas_parent_cell] = "internal"
    cells["integrated_node_kind"] = supervision_kind.to_numpy()
    cells["integrated_path"] = supervision.astype(str).map(path_for)
    cells["integrated_root"] = cells["integrated_path"].str.split(";").str[0]
    cells["integrated_candidate_leaf_ids"] = supervision.astype(str).map(
        lambda node: ";".join(_descendant_leaves(node, children, leaf_memo))
    )
    cells["integrated_from_atlas_parent"] = atlas_parent_cell
    cells.to_csv(cells_path, index=False)

    mapping_rows = []
    taxonomy_kind: dict[str, str] = {}
    taxonomy_nodes_path = tables / "taxonomy_nodes.csv"
    if taxonomy_nodes_path.exists():
        taxonomy_nodes = pd.read_csv(taxonomy_nodes_path, dtype=str)
        if {"tree_node_id", "node_kind"}.issubset(taxonomy_nodes.columns):
            taxonomy_kind = taxonomy_nodes.set_index("tree_node_id")[
                "node_kind"
            ].astype(str).to_dict()
    for source_node, target_node in sorted(node_mapping.items()):
        mapping_rows.append({
            "source_node_id": source_node,
            "source_node_kind": taxonomy_kind.get(
                source_node,
                "aligned_leaf" if source_node.startswith("ALIGNED_") else "internal",
            ),
            "integrated_node_id": target_node,
            "mapping_reason": mapping_reason[source_node],
        })

    nodes.to_csv(tables / "integrated_hierarchy_nodes.csv", index=False)
    edges.to_csv(tables / "integrated_hierarchy_edges.csv", index=False)
    pd.DataFrame(mapping_rows).to_csv(
        tables / "integrated_hierarchy_mapping.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        tables / "integrated_hierarchy_audit.csv", index=False
    )
    forest = {"roots": roots, "children": children}
    (tables / "integrated_hierarchy_forest.json").write_text(
        json.dumps(forest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "policy": "resolution_atlas_skeleton_then_taxonomy_completion",
        "n_aligned_nodes": len(aligned_nodes),
        "n_atlas_edges": int(n_atlas_edges_kept),
        "n_atlas_cycle_edges_skipped": int(len(skipped_atlas_cycles)),
        "n_taxonomy_merges_input": int(len(merges)),
        "n_taxonomy_nodes_retained": int(len(retained_taxa)),
        "n_taxonomy_nodes_collapsed": int(
            sum(value == "collapsed_into_atlas_component" for value in mapping_reason.values())
        ),
        "n_integrated_nodes": int(len(nodes)),
        "n_integrated_edges": int(len(edges)),
        "n_roots": int(len(roots)),
        "synthetic_root_added": SYNTHETIC_ROOT in set(nodes["node_id"]),
        "n_atlas_parent_cells": int(atlas_parent_cell.sum()),
        # These columns describe the structural audit tree only. They are
        # intentionally not consumed by CITEpool training.
        "audit_node_key": "integrated_supervision_node_id",
        "audit_node_kind_key": "integrated_node_kind",
    }
    (tables / "integrated_hierarchy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cytofuse-run",
        type=Path,
        required=True,
        help="Completed current CytoFuse output directory containing tables/.",
    )
    args = parser.parse_args()
    print(json.dumps(build_integrated_hierarchy(args.cytofuse_run), indent=2))


if __name__ == "__main__":
    main()


__all__ = ["build_integrated_hierarchy", "main"]
