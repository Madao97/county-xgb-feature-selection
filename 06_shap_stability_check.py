# -*- coding: utf-8 -*-
"""
06_shap_stability_check.py  (CLI + I18N + full figures)

Goal
----
Overfitting-risk check by SHAP Top-Pct features:
- Load FULL-model OOF predictions (from a previous training run)
- Load SHAP mean(|SHAP|) ranking from that full model
- Pick Top-Pct features (with a minimum count) -> train a simplified model with LOYO
- Compare FULL vs SIMPL overall and per-year metrics
- Export: CSVs, simplified OOF, models, feature list, and full set of figures

Key Features
------------
1) Multiple candidate paths supported for full OOF / SHAP files / features-used file
2) Read dataset_meta.json for axis target name & unit (optional)
3) Robust alignment:
   - Try merge by shared key columns (and year if present) -> strict alignment
   - Fallback to order alignment (with warning)
4) Figures:
   - Overall density scatter + residual hist (FULL & SIMPL)
   - Per-year: (FULL & SIMPL) scatter + residual + details CSV
   - Yearly comparison lines: R², wMAPE
   - Yearly difference bars: (SIMPL - FULL) for R², wMAPE (percentage points)

Dependencies
------------
pandas, numpy, xgboost, tqdm, matplotlib, scipy

Usage Example
-------------
python 06_shap_stability_check.py ^
  --base_dir D:\\SHANXI\\DATA\\train_dataset ^
  --work_dir D:\\SHANXI\\DATA\\train_dataset\\XGBoost_训练与解释_Pearson ^
  --out_dir  D:\\SHANXI\\DATA\\train_dataset\\XGBoost_训练与解释_Pearson\\过拟合风险验证_SHAPTop20 ^
  --lang cn ^
  --top_pct 0.20 --min_feats 10 ^
  --full_oof "OOF_预测结果_含校准.csv|OOF_预测结果.csv" ^
  --shap_mean_abs "SHAP_特征均值绝对值.csv|SHAP_mean_abs_all.csv" ^
  --features_used "使用特征清单.txt|features_used.txt" ^
  --save_models

Notes
-----
- By default, reads:
    X.csv, X_with_id.csv, y.csv from base_dir
  You can override by --x_csv --xid_csv --y_csv
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

from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import gaussian_filter


# =========================
# I18N
# =========================
I18N = {
    "cn": {
        "fig_dir": "图件",
        "year_dir": "按年分析",
        "model_dir": "模型_简化",
        "overall_cmp_csv": "对比_总体_全量_vs_简化.csv",
        "yearly_cmp_csv": "对比_逐年_全量_vs_简化.csv",
        "oof_simpl_csv": "OOF_简化模型_预测结果.csv",
        "feat_list_txt": "简化模型_特征清单.txt",
        "scatter_overall_full": "散点密度_总体_OOF_全量.png",
        "resid_overall_full": "残差直方图_总体_OOF_全量.png",
        "scatter_overall_simpl": "散点密度_总体_OOF_简化.png",
        "resid_overall_simpl": "残差直方图_总体_OOF_简化.png",
        "yearly_r2_cmp": "对比_按年_R2.png",
        "yearly_wmape_cmp": "对比_按年_wMAPE.png",
        "diff_r2_bar": "差值_按年_R2_简化减全量.png",
        "diff_wmape_bar": "差值_按年_wMAPE_简化减全量.png",
        "title_overall_full": "总体（全量）",
        "title_overall_simpl": "总体（简化Top{pct}%）",
        "title_year_full": "{year} 年（全量）",
        "title_year_simpl": "{year} 年（简化）",
        "xlabel_true": "真实值（{target}{unit}）",
        "ylabel_pred": "预测值（{target}{unit}）",
        "cbar": "栅格计数",
        "res_xlabel": "残差 (y_true - y_pred)",
        "res_ylabel": "频数",
        "legend_full": "全量",
        "legend_simpl": "简化(Top{pct}%)",
        "metric_r2": "R²",
        "metric_wmape": "wMAPE (%)",
        "plot_year": "年份",
        "done": "完成『过拟合风险验证（SHAP Top-Pct）』",
        "warn_fallback": "⚠️ 已回退为『顺序对齐』，请确认两个 OOF 的样本顺序一致。",
        "ok_merge": "✅ FULL OOF 与简化样本通过『主键（含年份）』成功对齐。",
    },
    "en": {
        "fig_dir": "figures",
        "year_dir": "by_year",
        "model_dir": "models_simpl",
        "overall_cmp_csv": "compare_overall_full_vs_simpl.csv",
        "yearly_cmp_csv": "compare_yearly_full_vs_simpl.csv",
        "oof_simpl_csv": "oof_simpl_predictions.csv",
        "feat_list_txt": "simpl_features.txt",
        "scatter_overall_full": "scatter_overall_full.png",
        "resid_overall_full": "residual_hist_overall_full.png",
        "scatter_overall_simpl": "scatter_overall_simpl.png",
        "resid_overall_simpl": "residual_hist_overall_simpl.png",
        "yearly_r2_cmp": "compare_yearly_R2.png",
        "yearly_wmape_cmp": "compare_yearly_wMAPE.png",
        "diff_r2_bar": "diff_yearly_R2_simpl_minus_full.png",
        "diff_wmape_bar": "diff_yearly_wMAPE_simpl_minus_full.png",
        "title_overall_full": "Overall (Full)",
        "title_overall_simpl": "Overall (Simplified Top{pct}%)",
        "title_year_full": "Year {year} (Full)",
        "title_year_simpl": "Year {year} (Simplified)",
        "xlabel_true": "Observed ({target}{unit})",
        "ylabel_pred": "Predicted ({target}{unit})",
        "cbar": "Bin count",
        "res_xlabel": "Residual (y_true - y_pred)",
        "res_ylabel": "Count",
        "legend_full": "Full",
        "legend_simpl": "Simplified (Top{pct}%)",
        "metric_r2": "R²",
        "metric_wmape": "wMAPE (%)",
        "plot_year": "Year",
        "done": "Done: SHAP Top-Pct stability check",
        "warn_fallback": "⚠️ Fallback to order-alignment. Please ensure two OOF files share identical row order.",
        "ok_merge": "✅ FULL OOF aligned to simplified samples by shared keys (including year).",
    }
}


# =========================
# Helpers
# =========================
EPS = 1e-8

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def read_csv_auto(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")

def split_candidates(spec: Optional[str]) -> List[str]:
    """
    Accept:
      - None
      - "a.csv|b.csv|c.csv"
    """
    if not spec:
        return []
    parts = [s.strip() for s in spec.split("|") if s.strip()]
    return parts

def first_existing(paths: List[str], base_dir: Optional[str] = None) -> str:
    """
    Try each path; if relative and base_dir provided, try base_dir/path.
    """
    tried = []
    for p in paths:
        cand = p
        if base_dir and (not os.path.isabs(p)):
            cand = os.path.join(base_dir, p)
        tried.append(cand)
        if os.path.isfile(cand):
            print(f"✅ Using: {cand}")
            return cand
    raise FileNotFoundError("No file found among candidates:\n" + "\n".join(tried))

def coerce_numeric_df(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    for c in Xn.columns:
        if pd.api.types.is_numeric_dtype(Xn[c]):
            continue
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn

def load_meta(meta_path: str) -> Tuple[str, str]:
    target_name, target_unit = "目标变量", ""
    if os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            target_name = meta.get("target_name", target_name)
            unit = meta.get("target_unit", "")
            # Keep unit as-is, the label function will handle spacing/punctuation per language.
            target_unit = unit if unit else ""
        except Exception:
            pass
    return target_name, target_unit


# =========================
# Metrics
# =========================
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def wmape(y_true, y_pred, eps=EPS):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))

def smape(y_true, y_pred, eps=EPS):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
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
# Plot styling
# =========================
def setup_fonts(font_size: int = 14) -> str:
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    prefer_chain = ["Microsoft YaHei", "SimHei", "Source Han Sans CN", "Noto Sans CJK SC", "DejaVu Sans"]
    chain = [f for f in prefer_chain if f in installed] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": chain,
        "font.size": font_size,
        "axes.unicode_minus": False,
        "axes.linewidth": 1.2,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.major.size": 5, "ytick.major.size": 5,
    })
    return chain[0]

def build_white_jet():
    base = plt.get_cmap("jet", 256)
    colors = base(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    return LinearSegmentedColormap.from_list("white_jet", colors)

CMAP_WHITE_JET = build_white_jet()


# =========================
# Density scatter + residual hist
# =========================
def _axis_unit_text(lang: str, unit: str) -> str:
    if not unit:
        return ""
    # cn: use "，单位" style is handled by caller if desired; keep compact:
    # Here we append unit directly with a leading separator for readability.
    if lang == "cn":
        return f"，{unit}"
    return f" {unit}"

def density_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: str,
    lang: str,
    T: Dict[str, str],
    target_name: str,
    target_unit: str,
    title: str,
    bins: int,
):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.size == 0:
        return

    vmin = float(min(yt.min(), yp.min()))
    vmax = float(max(yt.max(), yp.max()))
    if vmin == vmax:
        vmax = vmin + 1.0

    H2d, _, _ = np.histogram2d(yt, yp, bins=bins, range=[[vmin, vmax], [vmin, vmax]])
    H2d = gaussian_filter(H2d, sigma=1.0)

    # Fit: yp = a * yt + b  -> text "y = a · x + b"
    a, b = np.polyfit(yt, yp, 1)
    met = compute_metrics(yt, yp)

    fig, ax = plt.subplots(figsize=(6.8, 6.8), facecolor="white")
    im = ax.imshow(
        H2d.T, origin="lower",
        extent=[vmin, vmax, vmin, vmax],
        cmap=CMAP_WHITE_JET,
        norm=Normalize(vmin=H2d.min(), vmax=H2d.max()),
    )

    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.2)
    xx = np.linspace(vmin, vmax, 200)
    ax.plot(xx, a * xx + b, "r-", lw=2)

    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    unit_txt = _axis_unit_text(lang, target_unit)
    ax.set_xlabel(T["xlabel_true"].format(target=target_name, unit=unit_txt))
    ax.set_ylabel(T["ylabel_pred"].format(target=target_name, unit=unit_txt))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(T["cbar"])

    lines = [
        f"y = {a:.4f} · x + {b:.3f}",
        f"R² = {met['r2']:.4f}",
        f"wMAPE = {met['wmape']*100:,.2f}%",
        f"sMAPE = {met['smape']*100:,.2f}%",
    ]
    ax.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left",
        linespacing=1.25,
        bbox=dict(facecolor="white", alpha=0.85, pad=4, edgecolor="none")
    )

    ax.set_title(title, pad=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def residual_hist(
    residual: np.ndarray,
    out_png: str,
    T: Dict[str, str],
    title_prefix: str,
):
    r = np.asarray(residual, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return
    _rmse = rmse(np.zeros_like(r), r)
    _mae = mae(np.zeros_like(r), r)

    plt.figure(figsize=(8, 5), facecolor="white")
    plt.hist(r, bins=60, alpha=0.85)
    plt.xlabel(T["res_xlabel"])
    plt.ylabel(T["res_ylabel"])
    plt.title(f"{title_prefix}RMSE = {_rmse:,.2f}    MAE = {_mae:,.2f}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Yearly comparison plots
# =========================
def plot_yearline_compare(
    df_full: pd.DataFrame,
    df_simpl: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_png: str,
    T: Dict[str, str],
    pct_label: int,
    as_percent: bool = False,
):
    if df_full.empty or df_simpl.empty:
        return
    df_full = df_full.sort_values("year").reset_index(drop=True)
    df_simpl = df_simpl.sort_values("year").reset_index(drop=True)

    years = sorted(set(df_full["year"].astype(int).tolist()) | set(df_simpl["year"].astype(int).tolist()))
    y_full = []
    y_simpl = []
    for y in years:
        v1 = df_full.loc[df_full["year"].astype(int) == y, metric]
        v2 = df_simpl.loc[df_simpl["year"].astype(int) == y, metric]
        v1 = float(v1.values[0]) if len(v1) else np.nan
        v2 = float(v2.values[0]) if len(v2) else np.nan
        y_full.append(v1 * (100.0 if as_percent else 1.0))
        y_simpl.append(v2 * (100.0 if as_percent else 1.0))

    plt.figure(figsize=(10, 5))
    plt.plot(years, y_full, marker="o", label=T["legend_full"])
    plt.plot(years, y_simpl, marker="o", label=T["legend_simpl"].format(pct=pct_label))
    plt.xlabel(T["plot_year"])
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} ({T['legend_simpl'].format(pct=pct_label)} vs {T['legend_full']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_yeardiff_bar(
    df_cmp: pd.DataFrame,
    col_full: str,
    col_simpl: str,
    ylabel: str,
    out_png: str,
    as_percent: bool = False,
):
    if df_cmp.empty:
        return
    df = df_cmp.sort_values("year").reset_index(drop=True)
    years = df["year"].astype(int).values
    diff = df[col_simpl].values - df[col_full].values
    if as_percent:
        diff = diff * 100.0  # percentage points
    plt.figure(figsize=(10, 5))
    plt.bar(years.astype(str), diff)
    plt.axhline(0, color="k", linewidth=1)
    plt.xlabel("Year" if "Year" in ylabel else "年份")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} (Simpl - Full)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# =========================
# Model training (LOYO)
# =========================
def impute_by_median(Xtr: pd.DataFrame, Xva: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med = Xtr.median(axis=0, skipna=True)
    Xtr2 = Xtr.replace([np.inf, -np.inf], np.nan).fillna(med)
    Xva2 = Xva.replace([np.inf, -np.inf], np.nan).fillna(med)
    return Xtr2, Xva2

def fit_one_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, seed: int) -> Tuple[XGBRegressor, np.ndarray]:
    model = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0,
        random_state=seed, n_jobs=-1
    )
    model.fit(X_tr, y_tr, verbose=False)
    return model, model.predict(X_va)


# =========================
# Data loading
# =========================
def load_train_tables(base_dir: str, x_csv: str, xid_csv: str, y_csv: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    X = read_csv_auto(x_csv)
    Xi = read_csv_auto(xid_csv)
    y_df = read_csv_auto(y_csv)
    if "target" not in y_df.columns:
        raise KeyError("y.csv must contain column 'target'.")
    y = y_df["target"].values

    if "年份" in Xi.columns:
        years = Xi["年份"].values
    elif "year" in Xi.columns:
        years = Xi["year"].values
    else:
        raise RuntimeError("X_with_id.csv must contain column '年份' or 'year'.")

    idcols = [c for c in ["县ID","省份","城市","县区","年份","cid","省","市","县","year"] if c in Xi.columns]
    if len(idcols) == 0:
        ids = pd.DataFrame({"year": years})
    else:
        ids = Xi[idcols].copy()
    return X, y, years, ids


# =========================
# Feature selection: SHAP Top-Pct
# =========================
def load_shap_mean_abs(shap_path: str) -> pd.DataFrame:
    df = read_csv_auto(shap_path)
    # normalize columns
    if "feature" not in df.columns:
        for cand in ["features", "Feature", "列名", "特征"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "feature"})
                break
    if "mean_abs_shap" not in df.columns:
        for cand in ["mean_abs", "mean_shap_abs", "平均|SHAP|", "mean_abs_SHAP"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "mean_abs_shap"})
                break
    if "feature" not in df.columns or "mean_abs_shap" not in df.columns:
        raise RuntimeError("SHAP mean file must contain columns: (feature, mean_abs_shap).")
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df

def load_features_used_txt(path: Optional[str]) -> Optional[List[str]]:
    if not path or (not os.path.isfile(path)):
        return None
    feats = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = str(line).strip()
            if s:
                feats.append(s)
    return feats if feats else None

def pick_shap_top_pct(
    shap_mean: pd.DataFrame,
    top_pct: float,
    min_feats: int,
    whitelist: Optional[List[str]] = None,
) -> List[str]:
    n_total = len(shap_mean)
    k = max(int(np.ceil(n_total * float(top_pct))), int(min_feats))
    top = shap_mean.head(k)["feature"].astype(str).tolist()
    if whitelist is not None:
        top = [c for c in top if c in set(whitelist)]
    return top


# =========================
# Alignment (FULL OOF vs simplified samples)
# =========================
def guess_year_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["年份", "year", "_year", "Year"]:
        if c in df.columns:
            return c
    return None

def align_full_to_simplified(
    full_oof: pd.DataFrame,
    ids_simpl: pd.DataFrame,
    years_simpl: np.ndarray,
    y_true_simpl: np.ndarray,
    key_priority: Optional[List[str]],
    T: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Return y_true_full_aligned, y_pred_full_aligned with the same row order as simplified sample set.
    If merge alignment fails, fallback to order alignment (truncate to min length).
    """
    # Basic extraction
    if "y_true" not in full_oof.columns or "y_pred" not in full_oof.columns:
        raise RuntimeError("FULL OOF file must contain columns: y_true, y_pred.")
    year_col_full = guess_year_col(full_oof)

    # Candidate key columns: intersection between ids_simpl and full_oof
    base_keys = [c for c in ids_simpl.columns if c in full_oof.columns]
    # If user provided key_priority, use that ordering first
    if key_priority:
        base_keys = [c for c in key_priority if c in base_keys] + [c for c in base_keys if c not in set(key_priority)]

    # Ensure year is included if available on both sides
    keys = base_keys.copy()
    if year_col_full and (year_col_full in ids_simpl.columns) and (year_col_full not in keys):
        keys.append(year_col_full)
    # If ids_simpl has year but with different name, try to handle:
    # - if full has "year" but ids has "年份", or vice versa
    if year_col_full:
        if ("年份" in ids_simpl.columns and year_col_full == "year" and "year" not in keys):
            # We'll add both, but merge needs same column names; so we unify with temp columns below.
            pass

    # Try robust merge by normalized key strings
    try:
        # Build left keys table
        left = ids_simpl.copy()
        if year_col_full and year_col_full not in left.columns:
            # try map from "年份"<->"year"
            if year_col_full == "year" and "年份" in left.columns:
                left["year"] = left["年份"]
            elif year_col_full == "年份" and "year" in left.columns:
                left["年份"] = left["year"]

        # Build right keys table
        right = full_oof.copy()
        if year_col_full and year_col_full not in right.columns:
            # should not happen because year_col_full derived from right
            pass

        # Determine actual merge keys (after normalization)
        merge_keys = [c for c in left.columns if c in right.columns]
        # Prefer keys that include year if possible
        if year_col_full and year_col_full in merge_keys:
            # keep year within merge_keys
            pass

        # Need at least one key to merge; otherwise fallback
        if len(merge_keys) >= 1:
            # Convert to string to avoid dtype mismatch
            L = left[merge_keys].astype(str).copy()
            R = right[merge_keys].astype(str).copy()
            R["_y_true_full"] = right["y_true"].values
            R["_y_pred_full"] = right["y_pred"].values

            merged = L.merge(R, on=merge_keys, how="left")
            if merged["_y_pred_full"].notna().all():
                print(T["ok_merge"])
                y_true_full = merged["_y_true_full"].values.astype(float)
                y_pred_full = merged["_y_pred_full"].values.astype(float)
                return y_true_full, y_pred_full, True
    except Exception:
        pass

    # Fallback: order alignment
    print(T["warn_fallback"])
    n = min(len(full_oof), len(y_true_simpl))
    return full_oof["y_true"].values[:n], full_oof["y_pred"].values[:n], False


# =========================
# Main
# =========================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SHAP Top-Pct stability check (FULL vs SIMPL).")
    p.add_argument("--base_dir", required=True, help="Dataset base dir containing X.csv, X_with_id.csv, y.csv, dataset_meta.json")
    p.add_argument("--work_dir", required=True, help="Training output dir (contains FULL OOF, SHAP files, features_used)")
    p.add_argument("--out_dir", required=True, help="Output directory for this stability check")

    p.add_argument("--lang", default="cn", choices=["cn", "en"])
    p.add_argument("--top_pct", type=float, default=0.20)
    p.add_argument("--min_feats", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--density_bins", type=int, default=220)
    p.add_argument("--font_size", type=int, default=14)

    # candidate files: allow "a|b|c"
    p.add_argument("--full_oof", default="OOF_预测结果_含校准.csv|OOF_预测结果.csv", help="Candidate full OOF filenames (relative to work_dir) separated by |")
    p.add_argument("--shap_mean_abs", default="SHAP_特征均值绝对值.csv|SHAP_mean_abs_all.csv|SHAP_mean_abs.csv", help="Candidate SHAP mean-abs filenames (relative to work_dir) separated by |")
    p.add_argument("--features_used", default="使用特征清单.txt|features_used.txt|features_used.csv", help="Candidate features-used file (relative to work_dir) separated by |")

    # dataset file overrides
    p.add_argument("--x_csv", default=None)
    p.add_argument("--xid_csv", default=None)
    p.add_argument("--y_csv", default=None)

    p.add_argument("--save_models", action="store_true", help="Save fold models xgb_simpl_fold_*.json")
    p.add_argument("--key_priority", default=None, help="Comma-separated key column names to prefer for alignment (e.g., 县ID,年份,cid,year)")
    return p

def main():
    args = build_parser().parse_args()
    T = I18N[args.lang]

    t0 = time.time()
    font_used = setup_fonts(args.font_size)

    base_dir = args.base_dir
    work_dir = args.work_dir
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(os.path.join(out_dir, T["fig_dir"]))
    year_dir = ensure_dir(os.path.join(fig_dir, T["year_dir"]))
    model_dir = ensure_dir(os.path.join(out_dir, T["model_dir"])) if args.save_models else None

    # Meta
    target_name, target_unit = load_meta(os.path.join(base_dir, "dataset_meta.json"))

    # Dataset paths
    x_csv = args.x_csv if args.x_csv else os.path.join(base_dir, "X.csv")
    xid_csv = args.xid_csv if args.xid_csv else os.path.join(base_dir, "X_with_id.csv")
    y_csv = args.y_csv if args.y_csv else os.path.join(base_dir, "y.csv")

    # Resolve candidate files under work_dir
    full_oof_path = first_existing(split_candidates(args.full_oof), base_dir=work_dir)
    shap_path = first_existing(split_candidates(args.shap_mean_abs), base_dir=work_dir)

    feats_used_path = None
    # features_used may be a txt or csv; we try all candidates and accept the first existing
    try:
        feats_used_path = first_existing(split_candidates(args.features_used), base_dir=work_dir)
    except Exception:
        feats_used_path = None

    # Load full OOF and SHAP
    full_oof = read_csv_auto(full_oof_path)
    shap_mean = load_shap_mean_abs(shap_path)

    # Load features used whitelist (optional)
    whitelist = None
    if feats_used_path and os.path.isfile(feats_used_path):
        if feats_used_path.lower().endswith(".csv"):
            df = read_csv_auto(feats_used_path)
            col = "feature" if "feature" in df.columns else df.columns[0]
            whitelist = [c for c in df[col].dropna().astype(str).tolist()]
        else:
            whitelist = load_features_used_txt(feats_used_path)

    # Load training tables
    X_raw, y_raw, years_raw, ids_raw = load_train_tables(base_dir, x_csv, xid_csv, y_csv)

    # Pick top features from SHAP
    if whitelist is None:
        whitelist = list(X_raw.columns)
    top_feats = pick_shap_top_pct(shap_mean, args.top_pct, args.min_feats, whitelist=whitelist)
    if len(top_feats) == 0:
        raise RuntimeError("Top feature set is empty. Check SHAP file and features whitelist mapping.")

    with open(os.path.join(out_dir, T["feat_list_txt"]), "w", encoding="utf-8") as f:
        f.write("\n".join(top_feats))

    pct_label = int(round(args.top_pct * 100))
    print(f"✅ SHAP Top {pct_label}% features = {len(top_feats)}")
    print(f"🖋 Font used: {font_used}")
    print(f"📄 FULL OOF: {full_oof_path}")
    print(f"📄 SHAP mean abs: {shap_path}")

    # Prepare simplified feature matrix
    missing = [c for c in top_feats if c not in X_raw.columns]
    if missing:
        raise RuntimeError(f"Top features not found in X.csv: {missing[:20]} ...")

    Xs = X_raw[top_feats].copy()
    Xs = coerce_numeric_df(Xs)

    # Filter samples: finite y and finite features (after numeric conversion)
    mask = np.isfinite(y_raw)
    mask = mask & np.isfinite(Xs).all(axis=1).values
    Xs = Xs.loc[mask].reset_index(drop=True)
    y = y_raw[mask]
    years = np.asarray(years_raw)[mask]
    ids = ids_raw.loc[mask].reset_index(drop=True)

    # LOYO training for simplified model
    uniq_years = sorted(pd.unique(years))
    oof_pred = np.zeros(len(y), dtype=float)
    per_year_rows = []

    for yv in tqdm(uniq_years, desc="LOYO (simplified)"):
        tr_idx = np.where(years != yv)[0]
        va_idx = np.where(years == yv)[0]
        if tr_idx.size == 0 or va_idx.size == 0:
            continue

        Xtr = Xs.iloc[tr_idx, :]
        ytr = y[tr_idx]
        Xva = Xs.iloc[va_idx, :]
        yva = y[va_idx]

        Xtr2, Xva2 = impute_by_median(Xtr, Xva)
        model, pred = fit_one_fold(Xtr2, ytr, Xva2, seed=args.seed)
        oof_pred[va_idx] = pred

        met = compute_metrics(yva, pred)
        met["year"] = int(yv)
        per_year_rows.append(met)

        if model_dir is not None:
            model.save_model(os.path.join(model_dir, f"xgb_simpl_fold_{int(yv)}.json"))

    per_year_simpl = pd.DataFrame(per_year_rows).sort_values("year").reset_index(drop=True) if per_year_rows else \
                     pd.DataFrame(columns=["year","rmse","mae","r2","wmape","smape"])

    # Align FULL OOF to simplified samples
    key_priority = [s.strip() for s in args.key_priority.split(",")] if args.key_priority else None
    y_true_full, y_pred_full, merged_ok = align_full_to_simplified(
        full_oof=full_oof,
        ids_simpl=ids,
        years_simpl=years,
        y_true_simpl=y,
        key_priority=key_priority,
        T=T,
    )

    # If fallback, truncate simplified arrays to match full length
    if not merged_ok:
        n = min(len(y_true_full), len(y))
        y = y[:n]
        years = years[:n]
        ids = ids.iloc[:n, :].reset_index(drop=True)
        oof_pred = oof_pred[:n]
        y_true_full = y_true_full[:n]
        y_pred_full = y_pred_full[:n]

    # Overall metrics
    overall_full = compute_metrics(y_true_full, y_pred_full)
    overall_simpl = compute_metrics(y, oof_pred)

    overall_cmp = pd.DataFrame([
        {"model": T["legend_full"], "r2": overall_full["r2"], "wmape": overall_full["wmape"], "smape": overall_full["smape"],
         "rmse": overall_full["rmse"], "mae": overall_full["mae"], "n": int(len(y_true_full))},
        {"model": T["legend_simpl"].format(pct=pct_label), "r2": overall_simpl["r2"], "wmape": overall_simpl["wmape"], "smape": overall_simpl["smape"],
         "rmse": overall_simpl["rmse"], "mae": overall_simpl["mae"], "n": int(len(y))},
    ])
    overall_cmp.to_csv(os.path.join(out_dir, T["overall_cmp_csv"]), index=False, encoding="utf-8-sig")

    # Yearly metrics for full (computed from aligned arrays + years)
    df_full_local = pd.DataFrame({"y_true": y_true_full, "y_pred": y_pred_full, "year": years})
    rows_full = []
    for yv, sub in df_full_local.groupby("year"):
        met = compute_metrics(sub["y_true"].values, sub["y_pred"].values)
        met["year"] = int(yv)
        rows_full.append(met)
    per_year_full = pd.DataFrame(rows_full).sort_values("year").reset_index(drop=True) if rows_full else \
                    pd.DataFrame(columns=["year","rmse","mae","r2","wmape","smape"])

    # Yearly comparison table
    cmp_year = per_year_full.merge(per_year_simpl, on="year", suffixes=("_full", "_simpl"), how="outer").sort_values("year")
    # Keep a stable column order
    keep_cols = ["year",
                 "r2_full","wmape_full","smape_full","rmse_full","mae_full",
                 "r2_simpl","wmape_simpl","smape_simpl","rmse_simpl","mae_simpl"]
    for c in keep_cols:
        if c not in cmp_year.columns:
            cmp_year[c] = np.nan
    cmp_year = cmp_year[keep_cols]
    cmp_year.to_csv(os.path.join(out_dir, T["yearly_cmp_csv"]), index=False, encoding="utf-8-sig")

    # Export simplified OOF table
    oof_simpl = ids.copy()
    oof_simpl["y_true"] = y
    oof_simpl["y_pred_simpl"] = oof_pred
    oof_simpl["residual_simpl"] = oof_simpl["y_true"] - oof_simpl["y_pred_simpl"]
    oof_simpl["residual_pct_simpl"] = np.where(
        oof_simpl["y_true"] != 0, (oof_simpl["residual_simpl"] / oof_simpl["y_true"]) * 100.0, np.nan
    )
    oof_simpl.to_csv(os.path.join(out_dir, T["oof_simpl_csv"]), index=False, encoding="utf-8-sig")

    # =========================
    # Figures
    # =========================
    # Overall plots
    density_scatter(
        y_true_full, y_pred_full,
        out_png=os.path.join(fig_dir, T["scatter_overall_full"]),
        lang=args.lang, T=T, target_name=target_name, target_unit=target_unit,
        title=T["title_overall_full"], bins=args.density_bins
    )
    residual_hist(
        y_true_full - y_pred_full,
        out_png=os.path.join(fig_dir, T["resid_overall_full"]),
        T=T, title_prefix=f"{T['title_overall_full']}  "
    )

    density_scatter(
        y, oof_pred,
        out_png=os.path.join(fig_dir, T["scatter_overall_simpl"]),
        lang=args.lang, T=T, target_name=target_name, target_unit=target_unit,
        title=T["title_overall_simpl"].format(pct=pct_label), bins=args.density_bins
    )
    residual_hist(
        y - oof_pred,
        out_png=os.path.join(fig_dir, T["resid_overall_simpl"]),
        T=T, title_prefix=f"{T['title_overall_simpl'].format(pct=pct_label)}  "
    )

    # Per-year plots (both full and simpl)
    for yv in sorted(pd.unique(years).tolist()):
        yv_int = int(yv)
        yd = ensure_dir(os.path.join(year_dir, str(yv_int)))

        sub_full = df_full_local[df_full_local["year"] == yv]
        sub_simpl = pd.DataFrame({"y_true": y, "y_pred": oof_pred, "year": years})
        sub_simpl = sub_simpl[sub_simpl["year"] == yv]

        if len(sub_full):
            density_scatter(
                sub_full["y_true"].values, sub_full["y_pred"].values,
                out_png=os.path.join(yd, "scatter_full.png"),
                lang=args.lang, T=T, target_name=target_name, target_unit=target_unit,
                title=T["title_year_full"].format(year=yv_int), bins=args.density_bins
            )
            residual_hist(
                sub_full["y_true"].values - sub_full["y_pred"].values,
                out_png=os.path.join(yd, "residual_full.png"),
                T=T, title_prefix=f"{T['title_year_full'].format(year=yv_int)}  "
            )

        if len(sub_simpl):
            density_scatter(
                sub_simpl["y_true"].values, sub_simpl["y_pred"].values,
                out_png=os.path.join(yd, "scatter_simpl.png"),
                lang=args.lang, T=T, target_name=target_name, target_unit=target_unit,
                title=T["title_year_simpl"].format(year=yv_int), bins=args.density_bins
            )
            residual_hist(
                sub_simpl["y_true"].values - sub_simpl["y_pred"].values,
                out_png=os.path.join(yd, "residual_simpl.png"),
                T=T, title_prefix=f"{T['title_year_simpl'].format(year=yv_int)}  "
            )

        # Year details CSV (aligned)
        mask_y = (years.astype(int) == yv_int)
        ids_y = ids.loc[mask_y].copy().reset_index(drop=True)
        y_true_y = y[mask_y]
        y_pred_full_y = df_full_local.loc[df_full_local["year"].astype(int) == yv_int, "y_pred"].values
        y_pred_simpl_y = oof_pred[mask_y]

        # Ensure equal length for full pred in that year (should match if aligned)
        n_y = min(len(ids_y), len(y_true_y), len(y_pred_full_y), len(y_pred_simpl_y))
        ids_y = ids_y.iloc[:n_y, :].reset_index(drop=True)
        y_true_y = y_true_y[:n_y]
        y_pred_full_y = y_pred_full_y[:n_y]
        y_pred_simpl_y = y_pred_simpl_y[:n_y]

        details = ids_y.copy()
        details["y_true"] = y_true_y
        details["y_pred_full"] = y_pred_full_y
        details["y_pred_simpl"] = y_pred_simpl_y
        details["residual_full"] = details["y_true"] - details["y_pred_full"]
        details["residual_simpl"] = details["y_true"] - details["y_pred_simpl"]
        details.to_csv(os.path.join(yd, "details_full_vs_simpl.csv"), index=False, encoding="utf-8-sig")

    # Yearly compare line plots
    plot_yearline_compare(
        per_year_full, per_year_simpl,
        metric="r2", ylabel=T["metric_r2"],
        out_png=os.path.join(fig_dir, T["yearly_r2_cmp"]),
        T=T, pct_label=pct_label, as_percent=False
    )
    plot_yearline_compare(
        per_year_full, per_year_simpl,
        metric="wmape", ylabel=T["metric_wmape"],
        out_png=os.path.join(fig_dir, T["yearly_wmape_cmp"]),
        T=T, pct_label=pct_label, as_percent=True
    )

    # Yearly diff bars (simpl - full)
    plot_yeardiff_bar(
        cmp_year, "r2_full", "r2_simpl",
        ylabel="R² diff" if args.lang == "en" else "R²差值",
        out_png=os.path.join(fig_dir, T["diff_r2_bar"]),
        as_percent=False
    )
    plot_yeardiff_bar(
        cmp_year, "wmape_full", "wmape_simpl",
        ylabel="wMAPE diff (pp)" if args.lang == "en" else "wMAPE差值(百分点)",
        out_png=os.path.join(fig_dir, T["diff_wmape_bar"]),
        as_percent=True
    )

    # Summary
    print("\n✅", T["done"])
    print("Font:", font_used)
    print("Output:", out_dir)
    print(" -", T["overall_cmp_csv"])
    print(" -", T["yearly_cmp_csv"])
    print(" -", T["oof_simpl_csv"])
    print(" -", T["feat_list_txt"])
    if model_dir:
        print(" -", T["model_dir"], "/ xgb_simpl_fold_*.json")
    print(" -", T["fig_dir"], "/ ...")
    print(f"⏱️ {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
