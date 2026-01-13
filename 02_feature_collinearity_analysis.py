# -*- coding: utf-8 -*-
"""
02_feature_collinearity_analysis.py

Purpose
-------
Run a county-year regression workflow with:
1) Pearson-based redundancy pruning (feature-to-feature |corr| thresholding)
2) Leave-one-year-out cross-validation (LOYO)
3) Global fit plot with regression line and key metrics (R², wMAPE, sMAPE)
4) Optional SHAP attribution with contribution percentages (Top-K and grouped)

This script is designed to be shareable and path-agnostic. It does NOT assume
any local project directory layout.

Expected inputs
---------------
- X.csv: feature matrix (rows = samples, columns = features)
- y.csv: target table with column y_col (default: "target")
- id.csv (optional): sample identifiers, e.g., county name/id, year, etc.
  If provided, it will be merged into OOF output.
- year_col must exist in id.csv (preferred) or in X.csv. The LOYO split is based on year_col.

Optional inputs
---------------
- selected_features.csv: a list of initially selected features (one column named "feature")
- group_map.json: group name -> list of feature name prefixes, used for grouped SHAP summaries
- feature_groups.csv: a two-column mapping table: feature, group (overrides group_map.json)

Usage
-----
python 02_feature_collinearity_analysis.py ^
  --x data/X.csv ^
  --y data/y.csv ^
  --out_dir outputs/pearson_pruning ^
  --id_csv data/id.csv ^
  --year_col year ^
  --y_col target ^
  --selected_features selected_features.csv ^
  --corr_threshold 0.9 ^
  --prefer_rule target_corr ^
  --shap_topk 10 ^
  --shap_sample 3000 ^
  --group_map_json group_map.json

Outputs
-------
- metrics_overall.csv
- metrics_by_year.csv
- oof_predictions.csv
- pearson_corr_matrix.csv
- pearson_pruning_kept.csv / pearson_pruning_dropped.csv
- fit_scatter.png
- residual_hist.png
- shap_feature_importance.csv / shap_contribution_all.csv / shap_contribution_topk.csv
- shap_group_contribution.csv (if group info is provided)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBRegressor


# =========================
# Metrics
# =========================
EPS = 1e-8

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def wmape(y_true, y_pred, eps=EPS):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))

def smape(y_true, y_pred, eps=EPS):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)))

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


# =========================
# IO helpers
# =========================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")

def coerce_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    for c in Xn.columns:
        if pd.api.types.is_numeric_dtype(Xn[c]):
            continue
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    # Drop all-NaN columns
    keep_cols = [c for c in Xn.columns if not Xn[c].isna().all()]
    return Xn[keep_cols]

def drop_constant_columns(X: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for c in X.columns:
        s = X[c]
        if s.isna().all():
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return X[keep]


# =========================
# Feature selection inputs
# =========================
def load_selected_features(path: Optional[str], X_cols: List[str]) -> List[str]:
    if not path:
        return list(X_cols)
    if not os.path.isfile(path):
        return list(X_cols)
    df = read_csv(path)
    if "feature" not in df.columns:
        return list(X_cols)
    feats = [f for f in df["feature"].astype(str).tolist() if f in X_cols]
    return feats if len(feats) >= 5 else list(X_cols)


# =========================
# Group mapping (optional)
# =========================
def parse_group_map_json(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out = {}
    for k, v in obj.items():
        if isinstance(v, str):
            out[str(k)] = [v]
        else:
            out[str(k)] = [str(x) for x in v]
    return out

def feature_group(feature: str, group_map: Dict[str, List[str]]) -> str:
    s = str(feature)
    for gname, prefixes in group_map.items():
        for p in prefixes:
            if s.startswith(p):
                return gname
    return "Other"


# =========================
# Pearson redundancy pruning
# =========================
def safe_numeric_for_corr(X: pd.DataFrame) -> pd.DataFrame:
    Xn = coerce_numeric_df(X)
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    med = Xn.median(axis=0, skipna=True)
    return Xn.fillna(med)

def pearson_redundancy_pruning(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.9,
    prefer_rule: str = "target_corr",
    out_dir: Optional[str] = None,
) -> List[str]:
    """
    Prune highly correlated feature pairs with |corr| > threshold.
    Keep one feature per highly correlated pair:
      - prefer_rule="target_corr": keep the one with larger |corr(feature, y)|;
        break ties by higher variance.
      - prefer_rule="variance": keep the one with higher variance.
    """
    assert prefer_rule in ("target_corr", "variance")

    cols = list(X.columns)
    Xn = safe_numeric_for_corr(X[cols])

    # correlation with target (for tie-break)
    yser = pd.Series(y).astype(float)
    corr_y = {}
    for c in cols:
        try:
            corr_y[c] = float(pd.Series(Xn[c]).corr(yser, method="pearson"))
        except Exception:
            corr_y[c] = 0.0
    var_s = Xn.var(axis=0)

    corr = Xn.corr(method="pearson").abs()

    if out_dir:
        corr.to_csv(os.path.join(out_dir, "pearson_corr_matrix.csv"), encoding="utf-8-sig")

    # collect pairs above threshold (upper triangle)
    pairs = []
    arr = corr.values
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            v = arr[i, j]
            if np.isfinite(v) and v > threshold:
                pairs.append((v, cols[i], cols[j]))
    pairs.sort(reverse=True, key=lambda x: x[0])

    kept = set(cols)
    dropped = set()

    for v, c1, c2 in pairs:
        if c1 in dropped or c2 in dropped:
            continue

        if prefer_rule == "target_corr":
            s1, s2 = abs(corr_y.get(c1, 0.0)), abs(corr_y.get(c2, 0.0))
            if np.isclose(s1, s2):
                s1, s2 = float(var_s.get(c1, 0.0)), float(var_s.get(c2, 0.0))
        else:
            s1, s2 = float(var_s.get(c1, 0.0)), float(var_s.get(c2, 0.0))

        keep, drop = (c1, c2) if s1 >= s2 else (c2, c1)
        if drop in kept:
            kept.remove(drop)
        dropped.add(drop)

    kept_list = [c for c in cols if c in kept]
    dropped_list = [c for c in cols if c in dropped]

    if out_dir:
        pd.DataFrame({"kept_feature": kept_list}).to_csv(
            os.path.join(out_dir, "pearson_pruning_kept.csv"), index=False, encoding="utf-8-sig"
        )
        pd.DataFrame({"dropped_feature": dropped_list}).to_csv(
            os.path.join(out_dir, "pearson_pruning_dropped.csv"), index=False, encoding="utf-8-sig"
        )

    return kept_list


# =========================
# Modeling helpers
# =========================
def impute_median_train_valid(Xtr: pd.DataFrame, Xva: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med = Xtr.median(axis=0, skipna=True)
    Xtr2 = Xtr.replace([np.inf, -np.inf], np.nan).fillna(med)
    Xva2 = Xva.replace([np.inf, -np.inf], np.nan).fillna(med)
    return Xtr2, Xva2

def fit_one_fold(X_tr, y_tr, X_va, seed=42) -> Tuple[XGBRegressor, np.ndarray]:
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, verbose=False)
    y_hat = model.predict(X_va)
    return model, y_hat


# =========================
# Plots
# =========================
def plot_fit_scatter(y_true, y_pred, out_png: str, title: str = "") -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]

    # regression line: yp = a + b*yt
    b, a = np.polyfit(yt, yp, 1)
    met = compute_metrics(yt, yp)

    vmin = float(min(yt.min(), yp.min()))
    vmax = float(max(yt.max(), yp.max()))
    if vmin == vmax:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(6.8, 6.8), facecolor="white")
    ax.scatter(yt, yp, s=8, alpha=0.5)
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.2)
    xx = np.linspace(vmin, vmax, 200)
    ax.plot(xx, a + b * xx, linewidth=2)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    if title:
        ax.set_title(title)

    txt = "\n".join(
        [
            f"y_pred = {a:.3f} + {b:.4f} · y_obs",
            f"R² = {met['r2']:.4f}",
            f"wMAPE = {met['wmape'] * 100:.2f}%",
            f"sMAPE = {met['smape'] * 100:.2f}%",
        ]
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=4),
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_residual_hist(residual, out_png: str, title: str = "Residual histogram") -> None:
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.hist(residual, bins=60, alpha=0.85)
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# SHAP (optional)
# =========================
def shap_contributions(
    model: XGBRegressor,
    X_used: pd.DataFrame,
    out_dir: str,
    topk: int = 10,
    shap_sample: int = 3000,
    group_map_json: Optional[str] = None,
    feature_groups_csv: Optional[str] = None,
    seed: int = 42,
) -> None:
    """
    Compute mean(|SHAP|) and contribution percentages.
    Optionally aggregate by groups.
    """
    try:
        import shap  # noqa
    except Exception as e:
        with open(os.path.join(out_dir, "shap_status.txt"), "w", encoding="utf-8") as f:
            f.write(f"SHAP skipped: {repr(e)}\n")
        return

    rng = np.random.RandomState(seed)
    ns = min(int(shap_sample), X_used.shape[0])
    idx = rng.choice(X_used.index.values, size=ns, replace=False)
    Xs = X_used.loc[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_mean = (
        pd.DataFrame({"feature": X_used.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_mean.to_csv(os.path.join(out_dir, "shap_feature_importance.csv"), index=False, encoding="utf-8-sig")

    total = float(shap_mean["mean_abs_shap"].sum()) + EPS
    shap_mean["contribution_pct"] = shap_mean["mean_abs_shap"] / total * 100.0
    shap_mean.to_csv(os.path.join(out_dir, "shap_contribution_all.csv"), index=False, encoding="utf-8-sig")

    top = shap_mean.head(int(topk)).copy()
    top.to_csv(os.path.join(out_dir, f"shap_contribution_top{int(topk)}.csv"), index=False, encoding="utf-8-sig")

    # Group contributions
    group_map = parse_group_map_json(group_map_json)
    fg = None
    if feature_groups_csv and os.path.isfile(feature_groups_csv):
        fg = read_csv(feature_groups_csv)
        if "feature" in fg.columns and "group" in fg.columns:
            fg = fg[["feature", "group"]].copy()
        else:
            fg = None

    if fg is not None:
        shap_g = shap_mean.merge(fg, on="feature", how="left")
        shap_g["group"] = shap_g["group"].fillna("Other")
    elif group_map:
        shap_g = shap_mean.copy()
        shap_g["group"] = [feature_group(f, group_map) for f in shap_g["feature"]]
    else:
        shap_g = None

    if shap_g is not None:
        grp = (
            shap_g.groupby("group", as_index=False)["contribution_pct"]
            .sum()
            .sort_values("contribution_pct", ascending=False)
            .reset_index(drop=True)
        )
        grp.to_csv(os.path.join(out_dir, "shap_group_contribution.csv"), index=False, encoding="utf-8-sig")


# =========================
# CLI
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pearson pruning + LOYO XGBoost + SHAP analysis (shareable core code).")
    p.add_argument("--x", required=True, help="Path to X.csv")
    p.add_argument("--y", required=True, help="Path to y.csv")
    p.add_argument("--out_dir", required=True, help="Output directory")

    p.add_argument("--y_col", default="target", help="Target column name in y.csv")
    p.add_argument("--id_csv", default=None, help="Optional path to id.csv (sample identifiers)")
    p.add_argument("--year_col", default="year", help="Year column name (in id.csv if provided, else in X.csv)")

    p.add_argument("--selected_features", default=None, help="Optional CSV with column 'feature'")
    p.add_argument("--corr_threshold", type=float, default=0.9, help="Pearson |corr| threshold for pruning")
    p.add_argument("--prefer_rule", default="target_corr", choices=["target_corr", "variance"],
                   help="Which feature to keep when a highly correlated pair is found")

    p.add_argument("--model_seed", type=int, default=42, help="Random seed for model training")
    p.add_argument("--save_models", action="store_true", help="If set, save per-year models to out_dir/models/")

    p.add_argument("--run_shap", action="store_true", help="If set, compute SHAP contributions on a full refit model")
    p.add_argument("--shap_topk", type=int, default=10)
    p.add_argument("--shap_sample", type=int, default=3000)
    p.add_argument("--group_map_json", default=None, help="Optional group map JSON (group -> prefixes)")
    p.add_argument("--feature_groups_csv", default=None, help="Optional CSV (feature, group) for SHAP grouping")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    t0 = time.time()

    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))
    model_dir = ensure_dir(os.path.join(out_dir, "models")) if args.save_models else None

    # Load data
    X_raw = read_csv(args.x)
    y_df = read_csv(args.y)
    if args.y_col not in y_df.columns:
        raise KeyError(f"Target column '{args.y_col}' not found in y.csv.")
    y = y_df[args.y_col].values

    # Identify year column source
    ids = None
    if args.id_csv:
        ids = read_csv(args.id_csv)
        if args.year_col not in ids.columns:
            raise KeyError(f"Year column '{args.year_col}' not found in id.csv.")
        years = ids[args.year_col].values
    else:
        if args.year_col not in X_raw.columns:
            raise KeyError(
                f"Year column '{args.year_col}' not found. Provide --id_csv or include year_col in X.csv."
            )
        years = X_raw[args.year_col].values

    # Clean feature table: exclude year_col if it exists in X
    X_feat = X_raw.copy()
    if args.year_col in X_feat.columns:
        X_feat = X_feat.drop(columns=[args.year_col])

    X_feat = coerce_numeric_df(X_feat)
    X_feat = drop_constant_columns(X_feat)

    if X_feat.shape[0] != len(y) or len(y) != len(years):
        raise ValueError("Row mismatch among X, y, and year column. Ensure consistent row order and counts.")

    # Initial feature list (optional)
    initial_feats = load_selected_features(args.selected_features, list(X_feat.columns))
    X0 = X_feat[initial_feats].copy()

    # Pearson pruning
    kept_feats = pearson_redundancy_pruning(
        X0,
        y,
        threshold=float(args.corr_threshold),
        prefer_rule=args.prefer_rule,
        out_dir=out_dir,
    )
    X0 = X0[kept_feats].copy()

    # Remove invalid rows (finite y and features)
    X0 = X0.replace([np.inf, -np.inf], np.nan)
    mask = np.isfinite(y)
    # keep rows where all selected features are finite after coercion
    Xtmp = X0.apply(pd.to_numeric, errors="coerce")
    mask = mask & np.isfinite(Xtmp).all(axis=1).values

    X0 = X0.loc[mask].reset_index(drop=True)
    y2 = y[mask]
    years2 = np.asarray(years)[mask]

    if ids is not None:
        ids2 = ids.loc[mask].reset_index(drop=True)
    else:
        ids2 = pd.DataFrame({"year": years2})

    uniq_years = sorted(pd.unique(years2))
    print(f"LOYO folds (years): {uniq_years}")
    print(f"Final feature count after pruning: {X0.shape[1]}")

    # LOYO OOF
    oof_pred = np.zeros(len(y2), dtype=float)
    fold_rows = []

    for yv in tqdm(uniq_years, desc="LOYO CV"):
        tr_idx = np.where(years2 != yv)[0]
        va_idx = np.where(years2 == yv)[0]

        Xtr, ytr = X0.iloc[tr_idx, :], y2[tr_idx]
        Xva, yva = X0.iloc[va_idx, :], y2[va_idx]

        Xtr2, Xva2 = impute_median_train_valid(Xtr, Xva)
        model, pred = fit_one_fold(Xtr2, ytr, Xva2, seed=args.model_seed)
        oof_pred[va_idx] = pred

        met = compute_metrics(yva, pred)
        met["year"] = int(yv)
        fold_rows.append(met)

        if model_dir is not None:
            model.save_model(os.path.join(model_dir, f"xgb_fold_year_{int(yv)}.json"))

    overall = compute_metrics(y2, oof_pred)
    pd.DataFrame([overall]).to_csv(os.path.join(out_dir, "metrics_overall.csv"), index=False, encoding="utf-8-sig")

    by_year = pd.DataFrame(fold_rows).sort_values("year").reset_index(drop=True)
    by_year.to_csv(os.path.join(out_dir, "metrics_by_year.csv"), index=False, encoding="utf-8-sig")

    # OOF table
    oof_df = ids2.copy()
    oof_df["y_true"] = y2
    oof_df["y_pred"] = oof_pred
    oof_df["residual"] = oof_df["y_true"] - oof_df["y_pred"]
    oof_df.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False, encoding="utf-8-sig")

    # Plots
    plot_fit_scatter(y2, oof_pred, os.path.join(fig_dir, "fit_scatter.png"), title="OOF: Observed vs Predicted")
    plot_residual_hist(oof_df["residual"].values, os.path.join(fig_dir, "residual_hist.png"))

    # Full refit for SHAP (optional)
    if args.run_shap:
        X_imp = X0.replace([np.inf, -np.inf], np.nan)
        X_imp = X_imp.fillna(X_imp.median(axis=0, skipna=True))
        full_model = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=args.model_seed,
            n_jobs=-1,
        )
        full_model.fit(X_imp, y2, verbose=False)
        if model_dir is not None:
            full_model.save_model(os.path.join(model_dir, "xgb_full_refit.json"))

        shap_contributions(
            model=full_model,
            X_used=X_imp,
            out_dir=out_dir,
            topk=int(args.shap_topk),
            shap_sample=int(args.shap_sample),
            group_map_json=args.group_map_json,
            feature_groups_csv=args.feature_groups_csv,
            seed=args.model_seed,
        )

    # Save final feature list
    pd.DataFrame({"feature": list(X0.columns)}).to_csv(os.path.join(out_dir, "final_features.csv"),
                                                      index=False, encoding="utf-8-sig")

    print("Done. Output:", out_dir)
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
