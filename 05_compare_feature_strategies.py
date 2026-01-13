# -*- coding: utf-8 -*-
"""
05_compare_feature_strategies.py

Compare three feature strategies side-by-side:
  1) ALL      : use all features
  2) PEARSON  : initial candidate list (optional) + Pearson redundancy pruning
  3) RFE      : RFE recommended feature subset

For each mode, produce a full pipeline:
- LOYO (Leave-One-Year-Out) OOF predictions
- overall metrics + per-year metrics
- overall density scatter (fit + R² + wMAPE + sMAPE) + residual histogram
- per-year plots and per-year OOF CSV
- SHAP: mean |SHAP|, contribution percentage (TopK), optional grouped contribution + text summary

Additionally (global comparisons):
1) Side-by-side overall density scatter: compare_overall_scatter_3modes.png
2) Side-by-side per-year density scatter: compare_by_year/<YYYY>.png
3) Yearly metrics comparison: compare_yearly_wMAPE_3modes.png, compare_yearly_R2_3modes.png
4) Fit text format: y = a · x + b (left y, right a x + b); R² uses superscript ²
5) I18N: --lang cn / en

Inputs
------
Required:
- --x X.csv
- --y y.csv
- --out_dir output directory

Year information:
- If X.csv contains year_col, that's used
- Else provide --id_csv containing year_col

Optional feature lists:
- --pearson_selected_features: a CSV listing initial candidate features (col "feature")
- --rfe_features_csv: a CSV listing RFE recommended features (col "feature" or first column)

Optional SHAP grouping:
- --feature_groups_csv: columns ["feature","group"]

Outputs (under out_dir)
-----------------------
Per-mode directories:
  out_dir/<mode_name>/{figures,models,figures/by_year/...} + CSVs

Global compare outputs:
  out_dir/compare_overall_scatter_3modes.png
  out_dir/compare_by_year/<YYYY>.png
  out_dir/compare_yearly_wMAPE_3modes.png
  out_dir/compare_yearly_R2_3modes.png
  out_dir/compare_overall_metrics.csv

Usage example
-------------
python 05_compare_feature_strategies.py ^
  --x data/X.csv ^
  --y data/y.csv ^
  --out_dir outputs/compare_3modes ^
  --year_col year ^
  --id_csv data/id.csv ^
  --y_col target ^
  --lang en ^
  --pearson_selected_features outputs/corr/selected_features_pearson_global.csv ^
  --rfe_features_csv outputs/rfe/rfe_recommended_features.csv ^
  --corr_threshold 0.90 ^
  --prefer_rule target_corr ^
  --run_shap ^
  --shap_topk 10 ^
  --shap_sample 3000 ^
  --feature_groups_csv data/feature_groups.csv ^
  --save_models
"""

from __future__ import annotations

import argparse
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
from matplotlib.colors import LinearSegmentedColormap, Normalize

from xgboost import XGBRegressor
from scipy.ndimage import gaussian_filter


# =========================
# Basics
# =========================
EPS = 1e-8

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
    keep = [c for c in Xn.columns if not Xn[c].isna().all()]
    return Xn[keep]

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
# I18N
# =========================
I18N = {
    "cn": {
        "modes": {"ALL": "全量", "PEARSON": "Pearson去冗余", "RFE": "RFE推荐"},
        "dirs": {"fig": "图件", "model": "模型", "year": "按年分析", "year_cmp": "对比_按年"},
        "overall_scatter_cmp": "对比_总体散点_三路.png",
        "overall_metric_cmp": "对比_总体指标.csv",
        "peryear_wmape_cmp": "对比_按年_wMAPE_三路.png",
        "peryear_r2_cmp": "对比_按年_R2_三路.png",
        "scatter_title_overall": "总体（{mode}）",
        "xlabel": "真实值（{target}{unit}）",
        "ylabel": "预测值（{target}{unit}）",
        "cbar": "栅格计数",
        "eqn_label": "y = {a:.4f} · x + {b:.3f}",
        "r2_label": "R² = {r2:.4f}",
        "wmape_label": "wMAPE = {wmape:.2f}%",
        "smape_label": "sMAPE = {smape:.2f}%",
        "peryear_title": "{year} 年",
        "peryear_wmape": "按年 wMAPE",
        "peryear_r2": "按年 R²",
        "residual_xlabel": "残差 (y_true - y_pred)",
        "residual_ylabel": "频数",
        "done": "全部完成；输出目录：",
        "subdir": "子目录：",
        "oof_csv": "OOF_预测结果.csv",
        "year_metrics_csv": "按年指标汇总_扩展.csv",
        "overall_metrics_csv": "总体指标_扩展.csv",
        "shap_status": "SHAP_状态.txt",
    },
    "en": {
        "modes": {"ALL": "All", "PEARSON": "Pearson-pruned", "RFE": "RFE-recommended"},
        "dirs": {"fig": "figures", "model": "models", "year": "by_year", "year_cmp": "compare_by_year"},
        "overall_scatter_cmp": "compare_overall_scatter_3modes.png",
        "overall_metric_cmp": "compare_overall_metrics.csv",
        "peryear_wmape_cmp": "compare_yearly_wMAPE_3modes.png",
        "peryear_r2_cmp": "compare_yearly_R2_3modes.png",
        "scatter_title_overall": "Overall ({mode})",
        "xlabel": "Observed ({target}{unit})",
        "ylabel": "Predicted ({target}{unit})",
        "cbar": "Bin count",
        "eqn_label": "y = {a:.4f} · x + {b:.3f}",
        "r2_label": "R² = {r2:.4f}",
        "wmape_label": "wMAPE = {wmape:.2f}%",
        "smape_label": "sMAPE = {smape:.2f}%",
        "peryear_title": "Year {year}",
        "peryear_wmape": "Yearly wMAPE",
        "peryear_r2": "Yearly R²",
        "residual_xlabel": "Residual (y_true - y_pred)",
        "residual_ylabel": "Count",
        "done": "All done. Output at:",
        "subdir": "Subdir:",
        "oof_csv": "OOF_predictions.csv",
        "year_metrics_csv": "metrics_by_year.csv",
        "overall_metrics_csv": "metrics_overall.csv",
        "shap_status": "shap_status.txt",
    },
}


# =========================
# Metrics
# =========================
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def wmape(y_true, y_pred, eps=EPS):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))

def smape(y_true, y_pred, eps=EPS):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
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
# Plot style (white->jet)
# =========================
def build_white_jet():
    base = plt.get_cmap("jet", 256)
    colors = base(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    return LinearSegmentedColormap.from_list("white_jet", colors)

CMAP_WHITE_JET = build_white_jet()


# =========================
# Feature list readers
# =========================
def read_feature_list_csv(path: Optional[str], X_cols: List[str]) -> Optional[List[str]]:
    if not path or not os.path.isfile(path):
        return None
    df = read_csv(path)
    col = "feature" if "feature" in df.columns else df.columns[0]
    feats = [f for f in df[col].dropna().astype(str).tolist() if f in X_cols]
    return feats if len(feats) >= 1 else None


# =========================
# Pearson redundancy pruning
# =========================
def safe_numeric_for_corr(X: pd.DataFrame) -> pd.DataFrame:
    Xn = coerce_numeric_df(X)
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    med = Xn.median(axis=0, skipna=True)
    return Xn.fillna(med)

def pearson_prune(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float,
    prefer_rule: str,
    out_dir: str,
    save_prefix: str,
) -> List[str]:
    assert prefer_rule in ("target_corr", "variance")

    cols = list(X.columns)
    if len(cols) == 0:
        return cols

    Xn = safe_numeric_for_corr(X)
    yser = pd.Series(y).astype(float)

    corr_y = {}
    for c in cols:
        try:
            corr_y[c] = float(pd.Series(Xn[c]).corr(yser, method="pearson"))
        except Exception:
            corr_y[c] = 0.0
    var_s = Xn.var(axis=0)

    corr = Xn.corr(method="pearson").abs()
    corr.to_csv(os.path.join(out_dir, f"{save_prefix}_corr_matrix.csv"), encoding="utf-8-sig")

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

    for _, c1, c2 in pairs:
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

    pd.DataFrame({"kept_feature": kept_list}).to_csv(
        os.path.join(out_dir, f"{save_prefix}_kept.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"dropped_feature": dropped_list}).to_csv(
        os.path.join(out_dir, f"{save_prefix}_dropped.csv"), index=False, encoding="utf-8-sig"
    )
    return kept_list


# =========================
# Model
# =========================
def make_model(seed: int) -> XGBRegressor:
    return XGBRegressor(
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

def impute_median_train_valid(Xtr: pd.DataFrame, Xva: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med = Xtr.median(axis=0, skipna=True)
    Xtr2 = Xtr.replace([np.inf, -np.inf], np.nan).fillna(med)
    Xva2 = Xva.replace([np.inf, -np.inf], np.nan).fillna(med)
    return Xtr2, Xva2


# =========================
# Density scatter rendering (axes-reusable)
# =========================
def fit_for_text(yt: np.ndarray, yp: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    if len(yt) < 2:
        return 1.0, 0.0, {"r2": np.nan, "wmape": np.nan, "smape": np.nan}
    a, b = np.polyfit(yt, yp, 1)  # yp = a*yt + b
    met = compute_metrics(yt, yp)
    return a, b, met

def draw_density_scatter(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    T: Dict[str, str],
    target_name: str,
    target_unit: str,
    density_bins: int,
) -> Optional[object]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size == 0:
        ax.text(0.5, 0.5, "NA", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return None

    vmin = float(min(yt.min(), yp.min()))
    vmax = float(max(yt.max(), yp.max()))
    if vmin == vmax:
        vmax = vmin + 1.0

    H2d, _, _ = np.histogram2d(yt, yp, bins=density_bins, range=[[vmin, vmax], [vmin, vmax]])
    H2d = gaussian_filter(H2d, sigma=1.0)

    im = ax.imshow(
        H2d.T,
        origin="lower",
        extent=[vmin, vmax, vmin, vmax],
        cmap=CMAP_WHITE_JET,
        norm=Normalize(vmin=H2d.min(), vmax=H2d.max()),
    )

    # lines
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.2)
    a, b, met = fit_for_text(yt, yp)
    xx = np.linspace(vmin, vmax, 200)
    ax.plot(xx, a * xx + b, "r-", lw=2)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(T["xlabel"].format(target=target_name, unit=target_unit))
    ax.set_ylabel(T["ylabel"].format(target=target_name, unit=target_unit))
    ax.set_title(title, pad=6)

    lines = [
        T["eqn_label"].format(a=a, b=b),
        T["r2_label"].format(r2=met["r2"]),
        T["wmape_label"].format(wmape=met["wmape"] * 100 if np.isfinite(met["wmape"]) else np.nan),
        T["smape_label"].format(smape=met["smape"] * 100 if np.isfinite(met["smape"]) else np.nan),
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        linespacing=1.25,
        bbox=dict(facecolor="white", alpha=0.85, pad=4, edgecolor="none"),
    )
    return im

def save_density_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: str,
    title: str,
    T: Dict[str, str],
    target_name: str,
    target_unit: str,
    density_bins: int,
):
    fig, ax = plt.subplots(figsize=(6.8, 6.8), facecolor="white")
    im = draw_density_scatter(ax, y_true, y_pred, title, T, target_name, target_unit, density_bins)
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(T["cbar"])
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def save_residual_hist(
    residual: np.ndarray,
    out_png: str,
    T: Dict[str, str],
    title: str,
):
    residual = np.asarray(residual, float)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.hist(residual, bins=60, alpha=0.85)
    ax.set_xlabel(T["residual_xlabel"])
    ax.set_ylabel(T["residual_ylabel"])
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# SHAP (optional)
# =========================
def shap_outputs(
    model: XGBRegressor,
    X_used: pd.DataFrame,
    out_dir: str,
    topk: int,
    shap_sample: int,
    feature_groups_csv: Optional[str],
    seed: int,
    T: Dict[str, str],
):
    try:
        import shap  # noqa
    except Exception as e:
        with open(os.path.join(out_dir, T["shap_status"]), "w", encoding="utf-8") as f:
            f.write(f"SHAP skipped: {repr(e)}\n")
        return

    if X_used.shape[1] == 0 or X_used.shape[0] < 2:
        with open(os.path.join(out_dir, T["shap_status"]), "w", encoding="utf-8") as f:
            f.write("SHAP skipped: empty features or too few samples.\n")
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
    shap_mean.to_csv(os.path.join(out_dir, "shap_mean_abs.csv"), index=False, encoding="utf-8-sig")

    total = float(shap_mean["mean_abs_shap"].sum()) + EPS
    shap_mean["contribution_pct"] = shap_mean["mean_abs_shap"] / total * 100.0
    shap_mean.to_csv(os.path.join(out_dir, "shap_contrib_pct_all.csv"), index=False, encoding="utf-8-sig")

    top = shap_mean.head(int(topk)).copy()
    top.to_csv(os.path.join(out_dir, f"shap_contrib_pct_top{int(topk)}.csv"), index=False, encoding="utf-8-sig")

    # Plot TopK contribution percentages
    fig_dir = ensure_dir(os.path.join(out_dir, "figures" if "figures" in out_dir else "figures"))
    # But we always have per-mode figure directory; we will pass correct fig_dir from pipeline.
    # So keep this function purely file-based without plotting, plotting handled in pipeline if needed.

    # Grouped contribution (optional)
    if feature_groups_csv and os.path.isfile(feature_groups_csv):
        fg = read_csv(feature_groups_csv)
        if "feature" in fg.columns and "group" in fg.columns:
            fg = fg[["feature", "group"]].copy()
            shap_g = shap_mean.merge(fg, on="feature", how="left")
            shap_g["group"] = shap_g["group"].fillna("Other" if T is I18N["en"] else "其他")
            grp = (
                shap_g.groupby("group", as_index=False)["contribution_pct"]
                .sum()
                .sort_values("contribution_pct", ascending=False)
                .reset_index(drop=True)
            )
            grp.to_csv(os.path.join(out_dir, "shap_group_contrib_pct.csv"), index=False, encoding="utf-8-sig")

    # Text summary
    with open(os.path.join(out_dir, f"shap_top{int(topk)}_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Top SHAP contribution percentage\n" if T is I18N["en"] else "【SHAP 贡献百分比 Top】\n")
        cum = float(top["contribution_pct"].sum())
        for i, row in top.reset_index(drop=True).iterrows():
            f.write(f"{i+1}. {row['feature']}: {row['contribution_pct']:.1f}%\n")
        f.write(f"\nSum of Top{int(topk)}: {cum:.1f}%\n")


# =========================
# Per-year analysis
# =========================
def analyze_by_year(
    oof_df: pd.DataFrame,
    years: np.ndarray,
    mode_fig_dir: str,
    mode_year_dir: str,
    T: Dict[str, str],
    target_name: str,
    target_unit: str,
    density_bins: int,
) -> pd.DataFrame:
    tmp = oof_df.copy()
    tmp["_year"] = years
    rows = []
    for yv, sub in tmp.groupby("_year"):
        yt = sub["y_true"].values
        yp = sub["y_pred"].values
        res = yt - yp
        met = compute_metrics(yt, yp)
        met["year"] = int(yv)
        rows.append(met)

        yd = ensure_dir(os.path.join(mode_year_dir, str(int(yv))))
        save_density_scatter(
            yt,
            yp,
            os.path.join(yd, "scatter_true_vs_pred.png"),
            T["peryear_title"].format(year=int(yv)),
            T,
            target_name,
            target_unit,
            density_bins,
        )
        save_residual_hist(
            res,
            os.path.join(yd, "residual_hist.png"),
            T,
            title=f"{T['peryear_title'].format(year=int(yv))}",
        )
        sub.to_csv(os.path.join(yd, "oof_this_year.csv"), index=False, encoding="utf-8-sig")

    per_year_df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True) if rows else \
                  pd.DataFrame(columns=["year","rmse","mae","r2","wmape","smape"])

    # Small per-year plots for this mode
    if len(per_year_df):
        xs = per_year_df["year"].astype(str).tolist()

        plt.figure(figsize=(10, 5))
        plt.bar(xs, (per_year_df["wmape"] * 100.0).values)
        plt.ylabel("wMAPE (%)")
        plt.xlabel("Year" if T is I18N["en"] else "年份")
        plt.title(T["peryear_wmape"])
        plt.tight_layout()
        plt.savefig(os.path.join(mode_fig_dir, "yearly_wMAPE.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(xs, per_year_df["r2"].values, marker="o")
        plt.ylabel("R²")
        plt.xlabel("Year" if T is I18N["en"] else "年份")
        plt.title(T["peryear_r2"])
        plt.tight_layout()
        plt.savefig(os.path.join(mode_fig_dir, "yearly_R2.png"), dpi=300)
        plt.close()

    return per_year_df


# =========================
# Per-mode pipeline
# =========================
def run_mode_pipeline(
    mode_key: str,
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    years_all: np.ndarray,
    ids_all: pd.DataFrame,
    out_base: str,
    mode_name: str,
    T: Dict[str, str],
    target_name: str,
    target_unit: str,
    density_bins: int,
    corr_threshold: float,
    prefer_rule: str,
    pearson_selected_features: Optional[str],
    rfe_features_csv: Optional[str],
    save_models: bool,
    run_shap: bool,
    shap_topk: int,
    shap_sample: int,
    feature_groups_csv: Optional[str],
    seed: int,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]], Optional[pd.DataFrame]]:
    out_dir = ensure_dir(os.path.join(out_base, mode_name))
    fig_dir = ensure_dir(os.path.join(out_dir, T["dirs"]["fig"]))
    model_dir = ensure_dir(os.path.join(out_dir, T["dirs"]["model"])) if save_models else None
    year_root = ensure_dir(os.path.join(fig_dir, T["dirs"]["year"]))

    # Select features
    if mode_key == "ALL":
        feats = list(X_all.columns)
    elif mode_key == "PEARSON":
        feats0 = read_feature_list_csv(pearson_selected_features, list(X_all.columns))
        if feats0 is None:
            feats0 = list(X_all.columns)
        kept = pearson_prune(
            X_all[feats0], y_all,
            threshold=corr_threshold,
            prefer_rule=prefer_rule,
            out_dir=out_dir,
            save_prefix=f"pearson_prune_thr_{str(corr_threshold).replace('.','_')}",
        )
        feats = kept
    elif mode_key == "RFE":
        feats_rfe = read_feature_list_csv(rfe_features_csv, list(X_all.columns))
        if feats_rfe is None or len(feats_rfe) == 0:
            print(f"[{mode_key}] RFE feature list missing -> skip.")
            return None, None, None
        feats = feats_rfe
    else:
        raise ValueError("Unknown mode_key")

    if len(feats) == 0:
        print(f"[{mode_key}] No features -> skip.")
        return None, None, None

    X0 = X_all[feats].copy()

    # Clean rows: require finite y and numeric features
    X0 = X0.replace([np.inf, -np.inf], np.nan)
    Xnum = coerce_numeric_df(X0)
    Xnum = drop_constant_columns(Xnum)

    mask = np.isfinite(y_all)
    mask = mask & np.isfinite(Xnum).all(axis=1).values

    Xnum = Xnum.loc[mask].reset_index(drop=True)
    y = y_all[mask]
    years = np.asarray(years_all)[mask]
    ids = ids_all.loc[mask].reset_index(drop=True)

    pd.DataFrame({"feature": list(Xnum.columns)}).to_csv(
        os.path.join(out_dir, "features_used.csv"), index=False, encoding="utf-8-sig"
    )

    uniq_years = sorted(pd.unique(years))
    print(f"\n=== Mode={mode_name} | features={Xnum.shape[1]} | n={len(y)} | years={len(uniq_years)} ===")

    # LOYO CV
    oof_pred = np.zeros(len(y), dtype=float)
    per_fold_rows = []

    for yv in tqdm(uniq_years, desc=f"LOYO-{mode_name}"):
        tr_idx = np.where(years != yv)[0]
        va_idx = np.where(years == yv)[0]
        if tr_idx.size == 0 or va_idx.size == 0:
            continue

        Xtr = Xnum.iloc[tr_idx, :]
        ytr = y[tr_idx]
        Xva = Xnum.iloc[va_idx, :]
        yva = y[va_idx]

        Xtr2, Xva2 = impute_median_train_valid(Xtr, Xva)
        model = make_model(seed=seed)
        model.fit(Xtr2, ytr, verbose=False)
        pred = model.predict(Xva2)
        oof_pred[va_idx] = pred

        met = compute_metrics(yva, pred)
        met["year"] = int(yv)
        per_fold_rows.append(met)

        if model_dir is not None:
            model.save_model(os.path.join(model_dir, f"xgb_fold_{int(yv)}.json"))

    overall = compute_metrics(y, oof_pred)
    pd.DataFrame([overall]).to_csv(os.path.join(out_dir, T["overall_metrics_csv"]), index=False, encoding="utf-8-sig")

    per_year_df = pd.DataFrame(per_fold_rows).sort_values("year").reset_index(drop=True) if per_fold_rows else \
                  pd.DataFrame(columns=["year","rmse","mae","r2","wmape","smape"])
    per_year_df.to_csv(os.path.join(out_dir, T["year_metrics_csv"]), index=False, encoding="utf-8-sig")

    # OOF table
    oof_df = ids.copy()
    oof_df["y_true"] = y
    oof_df["y_pred"] = oof_pred
    oof_df["residual"] = oof_df["y_true"] - oof_df["y_pred"]
    oof_df["residual_pct"] = np.where(oof_df["y_true"] != 0, (oof_df["residual"] / oof_df["y_true"]) * 100, np.nan)
    oof_df["_year"] = years
    oof_df.to_csv(os.path.join(out_dir, T["oof_csv"]), index=False, encoding="utf-8-sig")

    # Overall plots
    save_density_scatter(
        oof_df["y_true"].values,
        oof_df["y_pred"].values,
        os.path.join(fig_dir, "scatter_overall.png"),
        T["scatter_title_overall"].format(mode=mode_name),
        T,
        target_name,
        target_unit,
        density_bins,
    )
    save_residual_hist(
        oof_df["residual"].values,
        os.path.join(fig_dir, "residual_hist_overall.png"),
        T,
        title="Residual histogram" if T is I18N["en"] else "残差直方图",
    )

    # Per-year plots + small plots
    per_year_df2 = analyze_by_year(
        oof_df.copy(),
        years,
        fig_dir,
        year_root,
        T,
        target_name,
        target_unit,
        density_bins,
    )
    # keep the one produced by analyze_by_year (same structure)
    per_year_df = per_year_df2
    per_year_df.to_csv(os.path.join(out_dir, T["year_metrics_csv"]), index=False, encoding="utf-8-sig")

    # Full refit for SHAP
    if run_shap and Xnum.shape[1] > 0 and Xnum.shape[0] > 1:
        X_imp = Xnum.replace([np.inf, -np.inf], np.nan)
        X_imp = X_imp.fillna(X_imp.median(axis=0, skipna=True))

        full_model = make_model(seed=seed)
        full_model.fit(X_imp, y, verbose=False)

        if model_dir is not None:
            full_model.save_model(os.path.join(model_dir, "xgb_full.json"))

        shap_outputs(
            model=full_model,
            X_used=X_imp,
            out_dir=out_dir,
            topk=shap_topk,
            shap_sample=shap_sample,
            feature_groups_csv=feature_groups_csv,
            seed=seed,
            T=T,
        )

        # Plot TopK bars inside this mode figure dir
        shap_top_csv = os.path.join(out_dir, f"shap_contrib_pct_top{int(shap_topk)}.csv")
        if os.path.isfile(shap_top_csv):
            top = read_csv(shap_top_csv)
            plt.figure(figsize=(9, 6))
            plt.barh(top["feature"][::-1], top["contribution_pct"][::-1])
            for i, v in enumerate(top["contribution_pct"][::-1].values):
                plt.text(v, i, f" {v:.1f}%", va="center")
            plt.xlabel("%")
            plt.ylabel("Feature" if T is I18N["en"] else "特征")
            ttl = f"Top {int(shap_topk)} SHAP contributions (sum=100%)" if T is I18N["en"] else f"前{int(shap_topk)} SHAP 贡献（总和=100%）"
            plt.title(ttl)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"shap_contrib_top{int(shap_topk)}.png"), dpi=300)
            plt.close()

        grp_csv = os.path.join(out_dir, "shap_group_contrib_pct.csv")
        if os.path.isfile(grp_csv):
            grp = read_csv(grp_csv)
            plt.figure(figsize=(9, 6))
            plt.barh(grp["group"][::-1], grp["contribution_pct"][::-1])
            for i, v in enumerate(grp["contribution_pct"][::-1].values):
                plt.text(v, i, f" {v:.1f}%", va="center")
            plt.xlabel("%")
            plt.ylabel("Group" if T is I18N["en"] else "特征组")
            ttl = "Group-wise SHAP contributions (sum=100%)" if T is I18N["en"] else "特征组别 SHAP 贡献（总和=100%）"
            plt.title(ttl)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "shap_group_contrib.png"), dpi=300)
            plt.close()

    return oof_df, overall, per_year_df


# =========================
# Global comparisons
# =========================
def save_overall_side_by_side(
    oof_all: Optional[pd.DataFrame],
    oof_pearson: Optional[pd.DataFrame],
    oof_rfe: Optional[pd.DataFrame],
    out_png: str,
    T: Dict[str, str],
    target_name: str,
    target_unit: str,
    density_bins: int,
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))
    triples = [
        (axes[0], T["modes"]["ALL"], oof_all),
        (axes[1], T["modes"]["PEARSON"], oof_pearson),
        (axes[2], T["modes"]["RFE"], oof_rfe),
    ]
    for ax, title, df in triples:
        if df is None or df.empty:
            ax.text(0.5, 0.5, f"{title}\n(NA)", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        im = draw_density_scatter(
            ax,
            df["y_true"].values,
            df["y_pred"].values,
            title=title,
            T=T,
            target_name=target_name,
            target_unit=target_unit,
            density_bins=density_bins,
        )
        if im is not None:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(T["cbar"])
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def save_yearly_side_by_side(
    oof_all: Optional[pd.DataFrame],
    oof_pearson: Optional[pd.DataFrame],
    oof_rfe: Optional[pd.DataFrame],
    out_dir: str,
    T: Dict[str, str],
    target_name: str,
    target_unit: str,
    density_bins: int,
):
    year_sets = []
    for df in [oof_all, oof_pearson, oof_rfe]:
        if df is not None and "_year" in df.columns:
            year_sets.append(set(df["_year"].dropna().tolist()))
    if not year_sets:
        return
    years_all = sorted(set().union(*year_sets))
    cmp_root = ensure_dir(os.path.join(out_dir, T["dirs"]["year_cmp"]))

    for yv in years_all:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))
        triples = [
            (axes[0], T["modes"]["ALL"], oof_all),
            (axes[1], T["modes"]["PEARSON"], oof_pearson),
            (axes[2], T["modes"]["RFE"], oof_rfe),
        ]
        for ax, title, df in triples:
            if df is None or df.empty:
                ax.text(0.5, 0.5, f"{title}\n(NA)", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            sub = df[df["_year"] == yv]
            if sub.empty:
                ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            im = draw_density_scatter(
                ax,
                sub["y_true"].values,
                sub["y_pred"].values,
                title=T["peryear_title"].format(year=int(yv)),
                T=T,
                target_name=target_name,
                target_unit=target_unit,
                density_bins=density_bins,
            )
            if im is not None:
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label(T["cbar"])
        plt.tight_layout()
        plt.savefig(os.path.join(cmp_root, f"{int(yv)}.png"), dpi=300, bbox_inches="tight")
        plt.close()

def save_yearly_metric_compare(
    per_year_all: Optional[pd.DataFrame],
    per_year_pearson: Optional[pd.DataFrame],
    per_year_rfe: Optional[pd.DataFrame],
    out_dir: str,
    T: Dict[str, str],
):
    def to_map(df):
        if df is None or df.empty:
            return {}
        return {int(r["year"]): r for _, r in df.iterrows()}

    mA = to_map(per_year_all)
    mP = to_map(per_year_pearson)
    mR = to_map(per_year_rfe)
    years = sorted(set(mA.keys()) | set(mP.keys()) | set(mR.keys()))
    if not years:
        return
    xs = [str(y) for y in years]

    # wMAPE
    yA = [mA.get(y, {}).get("wmape", np.nan) * 100 for y in years]
    yP = [mP.get(y, {}).get("wmape", np.nan) * 100 for y in years]
    yR = [mR.get(y, {}).get("wmape", np.nan) * 100 for y in years]
    plt.figure(figsize=(12, 5))
    plt.plot(xs, yA, marker="o", label=T["modes"]["ALL"])
    plt.plot(xs, yP, marker="o", label=T["modes"]["PEARSON"])
    plt.plot(xs, yR, marker="o", label=T["modes"]["RFE"])
    plt.ylabel("wMAPE (%)")
    plt.xlabel("Year" if T is I18N["en"] else "年份")
    plt.title(T["peryear_wmape"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, T["peryear_wmape_cmp"]), dpi=300)
    plt.close()

    # R²
    yA = [mA.get(y, {}).get("r2", np.nan) for y in years]
    yP = [mP.get(y, {}).get("r2", np.nan) for y in years]
    yR = [mR.get(y, {}).get("r2", np.nan) for y in years]
    plt.figure(figsize=(12, 5))
    plt.plot(xs, yA, marker="o", label=T["modes"]["ALL"])
    plt.plot(xs, yP, marker="o", label=T["modes"]["PEARSON"])
    plt.plot(xs, yR, marker="o", label=T["modes"]["RFE"])
    plt.ylabel("R²")
    plt.xlabel("Year" if T is I18N["en"] else "年份")
    plt.title(T["peryear_r2"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, T["peryear_r2_cmp"]), dpi=300)
    plt.close()


# =========================
# CLI
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare ALL / PEARSON / RFE feature strategies with LOYO + SHAP.")
    p.add_argument("--x", required=True, help="Path to X.csv")
    p.add_argument("--y", required=True, help="Path to y.csv")
    p.add_argument("--out_dir", required=True, help="Output directory")

    p.add_argument("--lang", default="cn", choices=["cn", "en"])
    p.add_argument("--y_col", default="target", help="Target column name in y.csv")
    p.add_argument("--id_csv", default=None, help="Optional id.csv containing year_col")
    p.add_argument("--year_col", default="year", help="Year column name for LOYO splitting (in id.csv or X.csv)")
    p.add_argument("--target_name", default=None, help="Optional target display name")
    p.add_argument("--target_unit", default="", help="Optional target unit (leading space handled by you)")

    p.add_argument("--pearson_selected_features", default=None, help="Optional Pearson selected feature list CSV")
    p.add_argument("--rfe_features_csv", default=None, help="Optional RFE recommended feature list CSV")

    p.add_argument("--corr_threshold", type=float, default=0.90)
    p.add_argument("--prefer_rule", default="target_corr", choices=["target_corr", "variance"])

    p.add_argument("--density_bins", type=int, default=220)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--save_models", action="store_true")
    p.add_argument("--run_shap", action="store_true")
    p.add_argument("--shap_topk", type=int, default=10)
    p.add_argument("--shap_sample", type=int, default=3000)
    p.add_argument("--feature_groups_csv", default=None)

    return p


# =========================
# Main
# =========================
def main():
    args = build_arg_parser().parse_args()
    t0 = time.time()

    T = I18N[args.lang]

    out_dir = ensure_dir(args.out_dir)

    # Load data
    X_raw = read_csv(args.x)
    y_df = read_csv(args.y)
    if args.y_col not in y_df.columns:
        raise KeyError(f"Target column '{args.y_col}' not found in y.csv.")
    y = y_df[args.y_col].values

    # Year source
    ids = None
    if args.id_csv:
        ids = read_csv(args.id_csv)
        if args.year_col not in ids.columns:
            raise KeyError(f"Year column '{args.year_col}' not found in id.csv.")
        years = ids[args.year_col].values
    else:
        if args.year_col not in X_raw.columns:
            raise KeyError(f"Year column '{args.year_col}' not found. Provide --id_csv or include year_col in X.csv.")
        years = X_raw[args.year_col].values

    # Build feature table
    X_feat = X_raw.copy()
    if args.year_col in X_feat.columns:
        X_feat = X_feat.drop(columns=[args.year_col])

    X_feat = coerce_numeric_df(X_feat)
    X_feat = drop_constant_columns(X_feat)

    if X_feat.shape[0] != len(y) or len(y) != len(years):
        raise ValueError("Row mismatch among X, y, and year column. Ensure consistent row order and counts.")

    # IDs fallback
    if ids is None:
        ids = pd.DataFrame({args.year_col: years})

    # Target display
    target_name = args.target_name if args.target_name else ("目标变量" if args.lang == "cn" else "Target")
    target_unit = args.target_unit if args.target_unit else ""

    # Mode names (directory names)
    mode_info = {
        "ALL": T["modes"]["ALL"],
        "PEARSON": T["modes"]["PEARSON"],
        "RFE": T["modes"]["RFE"],
    }

    # Run three pipelines
    oof_all, met_all, per_all = run_mode_pipeline(
        mode_key="ALL",
        X_all=X_feat, y_all=y, years_all=years, ids_all=ids,
        out_base=out_dir, mode_name=mode_info["ALL"],
        T=T, target_name=target_name, target_unit=target_unit,
        density_bins=int(args.density_bins),
        corr_threshold=float(args.corr_threshold),
        prefer_rule=str(args.prefer_rule),
        pearson_selected_features=args.pearson_selected_features,
        rfe_features_csv=args.rfe_features_csv,
        save_models=bool(args.save_models),
        run_shap=bool(args.run_shap),
        shap_topk=int(args.shap_topk),
        shap_sample=int(args.shap_sample),
        feature_groups_csv=args.feature_groups_csv,
        seed=int(args.seed),
    )

    oof_p, met_p, per_p = run_mode_pipeline(
        mode_key="PEARSON",
        X_all=X_feat, y_all=y, years_all=years, ids_all=ids,
        out_base=out_dir, mode_name=mode_info["PEARSON"],
        T=T, target_name=target_name, target_unit=target_unit,
        density_bins=int(args.density_bins),
        corr_threshold=float(args.corr_threshold),
        prefer_rule=str(args.prefer_rule),
        pearson_selected_features=args.pearson_selected_features,
        rfe_features_csv=args.rfe_features_csv,
        save_models=bool(args.save_models),
        run_shap=bool(args.run_shap),
        shap_topk=int(args.shap_topk),
        shap_sample=int(args.shap_sample),
        feature_groups_csv=args.feature_groups_csv,
        seed=int(args.seed),
    )

    oof_r, met_r, per_r = run_mode_pipeline(
        mode_key="RFE",
        X_all=X_feat, y_all=y, years_all=years, ids_all=ids,
        out_base=out_dir, mode_name=mode_info["RFE"],
        T=T, target_name=target_name, target_unit=target_unit,
        density_bins=int(args.density_bins),
        corr_threshold=float(args.corr_threshold),
        prefer_rule=str(args.prefer_rule),
        pearson_selected_features=args.pearson_selected_features,
        rfe_features_csv=args.rfe_features_csv,
        save_models=bool(args.save_models),
        run_shap=bool(args.run_shap),
        shap_topk=int(args.shap_topk),
        shap_sample=int(args.shap_sample),
        feature_groups_csv=args.feature_groups_csv,
        seed=int(args.seed),
    )

    # Global compare: overall scatter
    save_overall_side_by_side(
        oof_all, oof_p, oof_r,
        out_png=os.path.join(out_dir, T["overall_scatter_cmp"]),
        T=T,
        target_name=target_name,
        target_unit=target_unit,
        density_bins=int(args.density_bins),
    )

    # Global compare: per-year scatter
    save_yearly_side_by_side(
        oof_all, oof_p, oof_r,
        out_dir=out_dir,
        T=T,
        target_name=target_name,
        target_unit=target_unit,
        density_bins=int(args.density_bins),
    )

    # Global compare: yearly metrics curves
    save_yearly_metric_compare(per_all, per_p, per_r, out_dir=out_dir, T=T)

    # Global compare: overall metrics table
    rows = []
    if met_all is not None:
        rows.append({"mode": mode_info["ALL"], **met_all})
    if met_p is not None:
        rows.append({"mode": mode_info["PEARSON"], **met_p})
    if met_r is not None:
        rows.append({"mode": mode_info["RFE"], **met_r})

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, T["overall_metric_cmp"]), index=False, encoding="utf-8-sig")

    print("\n✅", T["done"], out_dir)
    print("   -", T["overall_scatter_cmp"])
    print("   -", T["peryear_wmape_cmp"])
    print("   -", T["peryear_r2_cmp"])
    for k, name in mode_info.items():
        print(f"   - [{name}] {T['subdir']} {os.path.join(out_dir, name)}")
    print(f"⏱️ {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
