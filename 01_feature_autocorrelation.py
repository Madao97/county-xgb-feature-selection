# -*- coding: utf-8 -*-
"""
01_feature_autocorrelation.py

Purpose
-------
Compute feature-to-feature correlation matrices (Pearson and Spearman) using all samples,
select Top-N features by absolute correlation with the target y, and export:
1) Heatmaps with axes labeled by 1..N indices (not raw feature names)
2) Two heatmap styles for each method:
   - Global ordering (sorted by |corr(feature, y)|)
   - Grouped ordering (block-sorted by feature source group, with separators)
3) A mapping table: index -> feature name -> group -> corr_with_y (Pearson/Spearman)

Expected inputs
---------------
- X.csv: feature matrix (rows = samples, columns = features)
- y.csv: target table containing a column (default: "target")
  The row order must match X.csv.

Usage
-----
python 01_feature_autocorrelation.py ^
  --x path/to/X.csv ^
  --y path/to/y.csv ^
  --out_dir outputs/corr_matrix ^
  --top_n 120 ^
  --y_col target ^
  --group_map_json group_map.json

group_map.json example
----------------------
{
  "Economy": ["ECON__"],
  "POI": ["POI__"],
  "Roads": ["OSM__"],
  "NDVI": ["NDVI__"],
  "NTL": ["NTL__"]
}

Notes
-----
- Spearman is implemented by rank-transforming X then computing Pearson on ranks.
- Non-numeric columns in X are ignored by default (converted to numeric when possible).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Plot settings (generic)
# =========================
DEFAULT_FONT_FAMILY = "Times New Roman"
DEFAULT_FONT_SIZE = 10

CMAP = "RdBu_r"
MAX_FIGSIZE_INCH = 40.0
CELL_INCH = 0.25


# =========================
# Helpers
# =========================
def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_csv(path: str) -> pd.DataFrame:
    # Try utf-8-sig first for Excel-saved CSVs; fall back to utf-8
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")


def coerce_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to numeric when possible; drop columns that become all-NaN.
    """
    Xn = X.copy()
    for c in Xn.columns:
        if pd.api.types.is_numeric_dtype(Xn[c]):
            continue
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    # Drop all-NaN columns
    keep_cols = [c for c in Xn.columns if not Xn[c].isna().all()]
    return Xn[keep_cols]


def drop_constant_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with zero variance (constant), which yield undefined correlations.
    """
    keep = []
    for c in X.columns:
        s = X[c]
        # treat all-NaN as drop; otherwise check nunique
        if s.isna().all():
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return X[keep]


def parse_group_map_json(path: Optional[str]) -> Dict[str, List[str]]:
    """
    Parse a JSON file mapping group name -> list of prefixes.
    If None, return empty dict.
    """
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # normalize to {str: [str,...]}
    out = {}
    for k, v in obj.items():
        if isinstance(v, str):
            out[str(k)] = [v]
        else:
            out[str(k)] = [str(x) for x in v]
    return out


def feature_group(feature_name: str, group_map: Dict[str, List[str]]) -> str:
    """
    Assign a feature to the first matching group based on prefix rules.
    If no rule matches, return "Other".
    """
    s = str(feature_name)
    for gname, prefixes in group_map.items():
        for p in prefixes:
            if s.startswith(p):
                return gname
    return "Other"


def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    return a.corr(b, method="pearson")


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    return a.corr(b, method="spearman")


def pick_topN_by_ycorr(
    X: pd.DataFrame,
    y: np.ndarray,
    top_n: int,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Rank features by |corr(feature, y)| and return Top-N.
    """
    yser = pd.Series(y)
    rows = []
    for c in X.columns:
        s = X[c]
        try:
            if method == "spearman":
                r = spearman_corr(s, yser)
            else:
                r = pearson_corr(s, yser)
        except Exception:
            r = np.nan
        rows.append((c, r))
    df = pd.DataFrame(rows, columns=["feature", "corr_with_y"]).dropna()
    df["abs_corr"] = df["corr_with_y"].abs()
    df = df.sort_values("abs_corr", ascending=False).head(int(top_n))
    return df.reset_index(drop=True)


def compute_corr_matrix(Xsub: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Compute NxN feature correlation matrix.
    - Pearson: X.corr(method="pearson")
    - Spearman: rank-transform then Pearson corr on ranks
    """
    if method == "spearman":
        return Xsub.rank(axis=0).corr(method="pearson")
    return Xsub.corr(method="pearson")


def fig_size_from_n(n: int) -> Tuple[float, float]:
    side = min(MAX_FIGSIZE_INCH, max(6.0, n * CELL_INCH))
    return (side, side)


def draw_heatmap_numbered(
    corr: pd.DataFrame,
    title: str,
    out_png: str,
    group_labels: Optional[List[str]] = None,
) -> None:
    """
    Draw heatmap using indices 1..N as tick labels.
    If group_labels is provided (length N), draw block separators.
    """
    n = corr.shape[0]
    idx_labels = np.arange(1, n + 1)

    plt.figure(figsize=fig_size_from_n(n), facecolor="white")
    ax = plt.gca()

    sns.heatmap(
        corr.values,
        cmap=CMAP,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlation"},
        xticklabels=idx_labels,
        yticklabels=idx_labels,
        linewidths=0.0,
        ax=ax,
    )

    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    ax.set_title(title, pad=10)

    # Reduce tick density for large N
    max_ticks = 60
    if n > max_ticks:
        step = math.ceil(n / max_ticks)
        for lbl in ax.get_xticklabels():
            i = int(lbl.get_text())
            lbl.set_visible(i % step == 1 or i == n)
        for lbl in ax.get_yticklabels():
            i = int(lbl.get_text())
            lbl.set_visible(i % step == 1 or i == n)
        plt.setp(ax.get_xticklabels(), rotation=0)

    # Draw group separators if requested
    if group_labels is not None and len(group_labels) == n:
        bounds = []
        cur = group_labels[0]
        for i, g in enumerate(group_labels):
            if g != cur:
                bounds.append(i)
                cur = g
        bounds.append(n)
        for b in bounds[:-1]:
            ax.axhline(b, color="k", lw=0.6)
            ax.axvline(b, color="k", lw=0.6)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def export_index_mapping(
    features: List[str],
    group_labels: List[str],
    pearson_with_y: Dict[str, float],
    spearman_with_y: Dict[str, float],
    out_csv: str,
) -> None:
    """
    Export index -> feature -> group -> correlations with y.
    """
    rows = []
    for i, f in enumerate(features, start=1):
        rows.append(
            {
                "idx": i,
                "feature": f,
                "group": group_labels[i - 1],
                "pearson_with_y": pearson_with_y.get(f, np.nan),
                "spearman_with_y": spearman_with_y.get(f, np.nan),
                "abs_corr_with_y": np.nanmax(
                    [abs(pearson_with_y.get(f, np.nan)), abs(spearman_with_y.get(f, np.nan))]
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


# =========================
# Main
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute feature autocorrelation heatmaps (Pearson/Spearman).")
    p.add_argument("--x", required=True, help="Path to X.csv (features).")
    p.add_argument("--y", required=True, help="Path to y.csv (target).")
    p.add_argument("--y_col", default="target", help="Target column name in y.csv. Default: target")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--top_n", type=int, default=120, help="Top-N features by |corr(feature, y)|.")
    p.add_argument("--group_map_json", default=None, help="Optional JSON file mapping group -> prefixes.")
    p.add_argument("--font_family", default=DEFAULT_FONT_FAMILY, help="Font family for plots.")
    p.add_argument("--font_size", type=int, default=DEFAULT_FONT_SIZE, help="Base font size for plots.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # Plot style
    plt.rcParams.update(
        {
            "font.family": args.font_family,
            "font.size": args.font_size,
            "axes.unicode_minus": True,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
        }
    )

    out_dir = ensure_out_dir(args.out_dir)
    group_map = parse_group_map_json(args.group_map_json)

    print("Loading X and y ...")
    X_raw = load_csv(args.x)
    y_df = load_csv(args.y)
    if args.y_col not in y_df.columns:
        raise KeyError(f"Target column '{args.y_col}' not found in {args.y}. Available: {list(y_df.columns)}")
    y = y_df[args.y_col].values

    # Coerce numeric and clean
    X = coerce_numeric_df(X_raw)
    X = drop_constant_columns(X)

    if X.shape[0] != len(y):
        raise ValueError(
            f"Row mismatch: X has {X.shape[0]} rows but y has {len(y)} rows. "
            "Ensure the row order and sample count match."
        )

    if X.shape[1] == 0:
        raise ValueError("No usable numeric feature columns found after cleaning.")

    # Compute Top-N by Pearson and Spearman
    methods = ["pearson", "spearman"]
    top_tables = {
        m: pick_topN_by_ycorr(X, y, args.top_n, method=m) for m in methods
    }

    # Pre-compute per-feature corr_with_y for both methods
    # (used for mapping export, even when selecting by one method)
    corr_with_y = {"pearson": {}, "spearman": {}}
    yser = pd.Series(y)
    for f in X.columns:
        s = X[f]
        try:
            corr_with_y["pearson"][f] = pearson_corr(s, yser)
        except Exception:
            corr_with_y["pearson"][f] = np.nan
        try:
            corr_with_y["spearman"][f] = spearman_corr(s, yser)
        except Exception:
            corr_with_y["spearman"][f] = np.nan

    for m in methods:
        df = top_tables[m].copy()
        feats_global = df["feature"].tolist()
        groups_global = [feature_group(f, group_map) for f in feats_global]

        # Export index mapping for global ordering
        export_index_mapping(
            features=feats_global,
            group_labels=groups_global,
            pearson_with_y=corr_with_y["pearson"],
            spearman_with_y=corr_with_y["spearman"],
            out_csv=os.path.join(out_dir, f"feature_index_{m}_global.csv"),
        )

        # Global heatmap
        corr_mat = compute_corr_matrix(X[feats_global], method=m)
        draw_heatmap_numbered(
            corr=corr_mat,
            title=f"Feature autocorrelation ({m.capitalize()}) (Top-{len(feats_global)} by |corr with y|)",
            out_png=os.path.join(out_dir, f"heatmap_{m}_topN_global_numbers.png"),
            group_labels=None,
        )

        # Grouped ordering: sort by group then by abs_corr (computed by selection method m)
        df["group"] = groups_global
        df_grouped = df.sort_values(["group", "abs_corr"], ascending=[True, False]).reset_index(drop=True)
        feats_grouped = df_grouped["feature"].tolist()
        groups_grouped = df_grouped["group"].tolist()

        # Export index mapping for grouped ordering
        export_index_mapping(
            features=feats_grouped,
            group_labels=groups_grouped,
            pearson_with_y=corr_with_y["pearson"],
            spearman_with_y=corr_with_y["spearman"],
            out_csv=os.path.join(out_dir, f"feature_index_{m}_grouped.csv"),
        )

        # Grouped heatmap with separators
        corr_mat_g = compute_corr_matrix(X[feats_grouped], method=m)
        draw_heatmap_numbered(
            corr=corr_mat_g,
            title=f"Feature autocorrelation ({m.capitalize()}, grouped) (Top-{len(feats_grouped)})",
            out_png=os.path.join(out_dir, f"heatmap_{m}_topN_grouped_numbers.png"),
            group_labels=groups_grouped if args.group_map_json else None,
        )

        # Export selected feature lists
        pd.DataFrame(
            {"idx": np.arange(1, len(feats_global) + 1), "feature": feats_global, "group": groups_global}
        ).to_csv(os.path.join(out_dir, f"selected_features_{m}_global.csv"), index=False, encoding="utf-8-sig")

        pd.DataFrame(
            {"idx": np.arange(1, len(feats_grouped) + 1), "feature": feats_grouped, "group": groups_grouped}
        ).to_csv(os.path.join(out_dir, f"selected_features_{m}_grouped.csv"), index=False, encoding="utf-8-sig")

    print("\nDone. Outputs written to:", out_dir)
    print("Generated files include:")
    print("  - heatmap_pearson_topN_global_numbers.png")
    print("  - heatmap_pearson_topN_grouped_numbers.png")
    print("  - heatmap_spearman_topN_global_numbers.png")
    print("  - heatmap_spearman_topN_grouped_numbers.png")
    print("  - feature_index_*_global.csv / feature_index_*_grouped.csv")
    print("  - selected_features_*_global.csv / selected_features_*_grouped.csv")


if __name__ == "__main__":
    main()
