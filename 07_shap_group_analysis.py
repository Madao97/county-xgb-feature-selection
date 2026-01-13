# -*- coding: utf-8 -*-
"""
07_shap_group_analysis.py  (Upgraded: CLI + robust feature lists + true CJK-safe + group compare)

What it does
------------
For three setups: ALL / PEARSON / RFE
- Load X, y from base_dir
- Decide feature subset per mode (robust, multi-candidate, with scan fallback)
- Train fresh XGB model on full data (median impute)
- Compute SHAP values on a sample
- Export:
  - model.json
  - used_features.txt
  - shap_importance.csv (mean_abs_shap + contribution_pct)
  - shap_importance_topK.png (contribution %)
  - shap_beeswarm.png (optional)
  - shap_dependence_*.png (optional, top features)
  - group_contrib.csv + group_contrib.png (if feature_groups.csv exists)
  - readme_top_features.txt

Extra:
- Cross-mode group comparison table:
    SHAP_group_compare_all_modes.csv

Notes
-----
- CJK-safe:
  If a CJK font exists, enforce it on all axes texts (including SHAP figures).
  If none, labels will be ASCII-sanitized to avoid tofu squares.

Usage
-----
python 07_shap_group_analysis.py ^
  --base_dir D:\\SHANXI\\DATA\\train_dataset ^
  --out_base D:\\SHANXI\\DATA\\train_dataset\\XGBoost_三路对比_26\\SHAP ^
  --pearson_list "CorrMatrix_numbered\\selected_features_pearson_global.csv|XGBoost_训练与解释_Pearson\\prune_pearson_thr_0_9_保留清单.csv" ^
  --rfe_list "XGBoost_RFE\\RFE_推荐子集_特征清单.csv" ^
  --save_npz 0 --do_beeswarm 1 --do_dependence 1

"""

from __future__ import annotations

import argparse
import json
import os
import re
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
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


# =========================
# CLI
# =========================
def build_parser():
    p = argparse.ArgumentParser("SHAP group analysis for ALL / PEARSON / RFE")
    p.add_argument("--base_dir", required=True)
    p.add_argument("--out_base", default=None, help="Output root; default: base_dir/XGBoost_三路对比_26/SHAP")

    p.add_argument("--x_csv", default=None)
    p.add_argument("--x_ascii_csv", default=None)
    p.add_argument("--y_csv", default=None)
    p.add_argument("--meta_json", default=None)
    p.add_argument("--feature_groups", default=None)

    # feature list candidates (use | separated paths, relative to base_dir allowed)
    p.add_argument("--pearson_list", default=None,
                   help="Candidate Pearson feature list files, separated by | (csv/txt).")
    p.add_argument("--rfe_list", default=None,
                   help="Candidate RFE feature list files, separated by | (csv/txt).")

    # scanning fallback (optional)
    p.add_argument("--scan_fallback", type=int, default=1, help="1: enable directory scan fallback")
    p.add_argument("--pearson_scan_dirs", default=None, help="Dirs to scan for Pearson lists (| separated)")
    p.add_argument("--rfe_scan_dirs", default=None, help="Dirs to scan for RFE lists (| separated)")

    # SHAP configs
    p.add_argument("--shap_topk", type=int, default=15)
    p.add_argument("--shap_beeswarm_top", type=int, default=30)
    p.add_argument("--shap_depend_top", type=int, default=6)
    p.add_argument("--shap_sample", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)

    # toggles
    p.add_argument("--save_npz", type=int, default=1, help="1: save shap_values.npz (can be large)")
    p.add_argument("--do_beeswarm", type=int, default=1)
    p.add_argument("--do_dependence", type=int, default=1)

    return p


# =========================
# Robust IO
# =========================
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def read_csv_robust(path: str) -> Optional[pd.DataFrame]:
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return None

def split_candidates(spec: Optional[str]) -> List[str]:
    if not spec:
        return []
    return [s.strip() for s in spec.split("|") if s.strip()]

def resolve_first_existing(cands: List[str], base_dir: str) -> Optional[str]:
    for p in cands:
        fp = p if os.path.isabs(p) else os.path.join(base_dir, p)
        if os.path.isfile(fp):
            return fp
    return None

def load_Xy(base_dir: str, x_csv: Optional[str], x_ascii: Optional[str], y_csv: Optional[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    X_path = (x_csv if x_csv else os.path.join(base_dir, "X.csv"))
    X_ascii = (x_ascii if x_ascii else os.path.join(base_dir, "X_ascii.csv"))
    Y_path = (y_csv if y_csv else os.path.join(base_dir, "y.csv"))

    xp = X_path if os.path.isfile(X_path) else X_ascii
    X = read_csv_robust(xp)
    if X is None:
        raise FileNotFoundError("Cannot read X.csv / X_ascii.csv")
    ydf = read_csv_robust(Y_path)
    if ydf is None or "target" not in ydf.columns:
        raise FileNotFoundError("y.csv must have column 'target'")

    # numeric conversion
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(ydf["target"], errors="coerce").values

    mask = np.isfinite(y)
    X, y = X.loc[mask].reset_index(drop=True), y[mask]
    return X, y


# =========================
# Font: Times + CJK fallback
# =========================
def setup_fonts() -> Tuple[str, Optional[str]]:
    serif_candidates = ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"]
    cjk_candidates = ["Microsoft YaHei", "SimHei", "Source Han Sans CN", "Noto Sans CJK SC",
                      "PingFang SC", "WenQuanYi Micro Hei"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    serif_main = next((n for n in serif_candidates if n in installed), "DejaVu Serif")
    cjk_font = next((n for n in cjk_candidates if n in installed), None)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [serif_main] + [n for n in serif_candidates if n != serif_main] + cjk_candidates,
        "axes.unicode_minus": False,
        "font.size": 12,
    })
    return serif_main, cjk_font

SERIF_MAIN, CJK_FALLBACK = setup_fonts()

def safe_label(s: str) -> str:
    if CJK_FALLBACK:
        return str(s)
    return re.sub(r"[^\w\s\-]+", "_", str(s))

def apply_cjk_axis(ax):
    if not CJK_FALLBACK:
        return
    items = [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    for t in items:
        try:
            t.set_fontname(CJK_FALLBACK)
        except Exception:
            pass

def apply_cjk_current_fig():
    """Apply CJK fallback to ALL axes in current figure (useful for SHAP plots)."""
    if not CJK_FALLBACK:
        return
    fig = plt.gcf()
    for ax in fig.get_axes():
        apply_cjk_axis(ax)


# =========================
# XGB training
# =========================
def impute_median(X: pd.DataFrame) -> pd.DataFrame:
    med = X.median(axis=0, skipna=True)
    X2 = X.replace([np.inf, -np.inf], np.nan).fillna(med)
    return X2

def train_xgb(X: pd.DataFrame, y: np.ndarray, seed: int) -> XGBRegressor:
    mdl = XGBRegressor(
        n_estimators=1200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0,
        random_state=seed, n_jobs=-1
    )
    mdl.fit(X, y, verbose=False)
    return mdl


# =========================
# Feature list discovery
# =========================
def read_feature_list_file(path: str) -> Optional[List[str]]:
    try:
        if path.lower().endswith(".txt"):
            feats = [ln.strip() for ln in open(path, "r", encoding="utf-8").read().splitlines() if ln.strip()]
            return feats if feats else None
        df = read_csv_robust(path)
        if df is None or df.empty:
            return None
        # prefer 'feature' column
        cols_low = [c.lower() for c in df.columns]
        if "feature" in cols_low:
            col = df.columns[cols_low.index("feature")]
        else:
            col = df.columns[0]
        feats = [str(x).strip() for x in df[col].dropna().tolist()]
        return feats if feats else None
    except Exception:
        return None

def scan_feature_list(
    dirs: List[str],
    name_hints: List[str],
) -> Optional[str]:
    cand = []
    for root in dirs:
        if not os.path.isdir(root):
            continue
        for cur, _, files in os.walk(root):
            cur_low = cur.lower()
            if not any(h.lower() in cur_low for h in name_hints):
                continue
            for fn in files:
                if not (fn.lower().endswith(".csv") or fn.lower().endswith(".txt")):
                    continue
                fp = os.path.join(cur, fn)
                score = 0
                fn_low = fn.lower()
                # heuristic scoring
                if ("保留" in fn) or ("kept" in fn_low) or ("selected" in fn_low):
                    score += 800
                if ("pearson" in fn_low) or ("prune" in fn_low):
                    score += 500
                if ("rfe" in fn_low) or ("推荐" in fn):
                    score += 500
                score += min(os.path.getsize(fp), 10_000_000)  # cap
                cand.append((score, fp))
    cand.sort(reverse=True, key=lambda x: x[0])
    return cand[0][1] if cand else None

def decide_features_for_mode(
    mode: str,
    all_columns: List[str],
    base_dir: str,
    pearson_cands: List[str],
    rfe_cands: List[str],
    scan_fallback: bool,
    pearson_scan_dirs: List[str],
    rfe_scan_dirs: List[str],
) -> Tuple[List[str], str]:
    """
    Return (features, source_note)
    """
    if mode == "ALL":
        return list(all_columns), "ALL columns"

    if mode == "PEARSON":
        fp = resolve_first_existing(pearson_cands, base_dir)
        if fp:
            feats = read_feature_list_file(fp)
            if feats:
                feats = [f for f in feats if f in all_columns]
                if len(feats) >= 2:
                    return feats, f"Pearson list: {fp}"
        if scan_fallback and pearson_scan_dirs:
            hit = scan_feature_list(pearson_scan_dirs, ["pearson", "prune", "保留", "kept", "selected"])
            if hit:
                feats = read_feature_list_file(hit)
                if feats:
                    feats = [f for f in feats if f in all_columns]
                    if len(feats) >= 2:
                        return feats, f"Pearson scan hit: {hit}"
        return list(all_columns), "Pearson list not found -> fallback ALL"

    if mode == "RFE":
        fp = resolve_first_existing(rfe_cands, base_dir)
        if fp:
            feats = read_feature_list_file(fp)
            if feats:
                feats = [f for f in feats if f in all_columns]
                if len(feats) >= 2:
                    return feats, f"RFE list: {fp}"
        if scan_fallback and rfe_scan_dirs:
            hit = scan_feature_list(rfe_scan_dirs, ["rfe", "推荐", "subset", "feature"])
            if hit:
                feats = read_feature_list_file(hit)
                if feats:
                    feats = [f for f in feats if f in all_columns]
                    if len(feats) >= 2:
                        return feats, f"RFE scan hit: {hit}"
        return list(all_columns), "RFE list not found -> fallback ALL"

    raise ValueError("Unknown mode.")


# =========================
# Plots
# =========================
def plot_importance(df_imp: pd.DataFrame, out_png: str, topk: int, title: str):
    top = df_imp.head(topk).copy()
    top["feature_plot"] = top["feature"].map(safe_label)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature_plot"][::-1], top["contribution_pct"][::-1], alpha=0.9)
    for i, v in enumerate(top["contribution_pct"][::-1].values):
        ax.text(v, i, f" {v:.1f}%", va="center")
    ax.set_xlabel("Contribution (%)")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    apply_cjk_axis(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_group_contrib(df_imp: pd.DataFrame, fg_path: str, out_dir: str):
    if not os.path.isfile(fg_path):
        return None

    fg = read_csv_robust(fg_path)
    if fg is None or fg.empty or ("feature" not in fg.columns) or ("group" not in fg.columns):
        return None

    g = df_imp.merge(fg[["feature", "group"]], on="feature", how="left").fillna({"group": "Others"})
    grp = g.groupby("group", as_index=False)["contribution_pct"].sum().sort_values("contribution_pct", ascending=False)
    grp["group_plot"] = grp["group"].map(safe_label)

    out_csv = os.path.join(out_dir, "group_contrib.csv")
    grp[["group", "contribution_pct"]].to_csv(out_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(grp["group_plot"][::-1], grp["contribution_pct"][::-1], alpha=0.9)
    for i, v in enumerate(grp["contribution_pct"][::-1].values):
        ax.text(v, i, f" {v:.1f}%", va="center")
    ax.set_xlabel("Contribution (%)")
    ax.set_ylabel("Feature group")
    ax.set_title("Grouped contribution (sum = 100%)")
    apply_cjk_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "group_contrib.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return grp

def beeswarm_and_dependence(shap_values: np.ndarray, Xs: pd.DataFrame, out_dir: str,
                            bsw_top: int, dep_top: int, do_beeswarm: bool, do_dependence: bool):
    import shap

    # Beeswarm
    if do_beeswarm:
        try:
            Xs_plot = Xs.copy()
            Xs_plot.columns = [safe_label(c) for c in Xs_plot.columns]
            shap.summary_plot(shap_values, Xs_plot, show=False, max_display=bsw_top)
            apply_cjk_current_fig()
            plt.gca().set_title("SHAP beeswarm (top features)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "shap_beeswarm.png"), dpi=300, bbox_inches="tight")
            plt.close()
        except Exception:
            plt.close()

    # Dependence
    if do_dependence:
        try:
            mean_abs = np.abs(shap_values).mean(axis=0)
            order = np.argsort(-mean_abs)
            top_idx = order[:min(dep_top, Xs.shape[1])]
            inter_col = Xs.columns[order[1]] if order.size > 1 else None

            for rank, j in enumerate(top_idx, start=1):
                feat = Xs.columns[j]
                feat_plot = safe_label(feat)
                shap.dependence_plot(feat, shap_values, Xs, interaction_index=inter_col, show=False)
                apply_cjk_current_fig()
                plt.gca().set_title(f"Dependence: {feat_plot}")
                plt.tight_layout()
                fname = re.sub(r"[^0-9A-Za-z_]+", "_", feat_plot)[:80]
                plt.savefig(os.path.join(out_dir, f"shap_dependence_{rank:02d}_{fname}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()
        except Exception:
            plt.close()

def write_top_readme(df_imp: pd.DataFrame, out_dir: str, topk: int):
    top = df_imp.head(topk).copy()
    cum = float(top["contribution_pct"].sum())
    path = os.path.join(out_dir, "readme_top_features.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Top{topk} feature contribution percentages (sum=100% over all features)\n\n")
        for i, r in enumerate(top.itertuples(index=False), start=1):
            f.write(f"{i:02d}. {r.feature}: {r.contribution_pct:.2f}%\n")
        f.write(f"\nCumulative Top{topk}: {cum:.2f}%\n")


# =========================
# One mode
# =========================
def run_one_mode(
    mode: str,
    X_full: pd.DataFrame,
    y: np.ndarray,
    feats: List[str],
    out_dir: str,
    seed: int,
    shap_sample: int,
    shap_topk: int,
    bsw_top: int,
    dep_top: int,
    save_npz: bool,
    do_beeswarm: bool,
    do_dependence: bool,
    fg_path: str,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    out_dir = ensure_dir(out_dir)

    X = X_full[feats].copy()
    X_imp = impute_median(X)

    model = train_xgb(X_imp, y, seed=seed)
    model.save_model(os.path.join(out_dir, "model.json"))
    # save feature list
    with open(os.path.join(out_dir, "used_features.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feats))

    # SHAP
    group_df = None
    try:
        import shap
        ns = min(int(shap_sample), X_imp.shape[0])
        rng = np.random.RandomState(seed)
        idx = rng.choice(X_imp.index.values, size=ns, replace=False)
        Xs = X_imp.loc[idx]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xs)

        if save_npz:
            np.savez_compressed(
                os.path.join(out_dir, "shap_values.npz"),
                shap_values=shap_values,
                columns=np.array(Xs.columns),
                index=np.array(Xs.index),
            )

        mean_abs = np.abs(shap_values).mean(axis=0)
        df_imp = pd.DataFrame({
            "feature": Xs.columns.astype(str),
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        total = float(df_imp["mean_abs_shap"].sum())
        df_imp["contribution_pct"] = np.where(total > 0, df_imp["mean_abs_shap"] / total * 100.0, 0.0)
        df_imp.to_csv(os.path.join(out_dir, "shap_importance.csv"), index=False, encoding="utf-8-sig")

        plot_importance(
            df_imp,
            os.path.join(out_dir, "shap_importance_topK.png"),
            topk=shap_topk,
            title=f"{mode}: Top features by contribution (%)"
        )

        beeswarm_and_dependence(
            shap_values, Xs, out_dir,
            bsw_top=bsw_top, dep_top=dep_top,
            do_beeswarm=do_beeswarm, do_dependence=do_dependence
        )

        group_df = plot_group_contrib(df_imp, fg_path=fg_path, out_dir=out_dir)
        write_top_readme(df_imp, out_dir, topk=shap_topk)

        return df_imp, group_df

    except Exception as e:
        with open(os.path.join(out_dir, "SHAP_status.txt"), "w", encoding="utf-8") as f:
            f.write(f"SHAP failed/skipped: {repr(e)}\n")
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "contribution_pct"]), None


# =========================
# Main
# =========================
def main():
    args = build_parser().parse_args()

    base_dir = args.base_dir
    out_base = args.out_base if args.out_base else os.path.join(base_dir, "XGBoost_三路对比_26", "SHAP")
    ensure_dir(out_base)

    # default scan dirs
    pearson_scan_dirs = split_candidates(args.pearson_scan_dirs) if args.pearson_scan_dirs else [
        os.path.join(base_dir, "XGBoost_训练与解释_Pearson"),
        os.path.join(base_dir, "XGBoost_训练与解释"),
        os.path.join(base_dir, "CorrMatrix_numbered"),
        base_dir,
    ]
    rfe_scan_dirs = split_candidates(args.rfe_scan_dirs) if args.rfe_scan_dirs else [
        os.path.join(base_dir, "XGBoost_RFE"),
        base_dir,
    ]

    # default feature list candidates
    pearson_cands = split_candidates(args.pearson_list) if args.pearson_list else [
        os.path.join("CorrMatrix_numbered", "selected_features_pearson_global.csv"),
        os.path.join("XGBoost_训练与解释_Pearson", "features_used.txt"),
        os.path.join("XGBoost_训练与解释_Pearson", "使用特征清单.txt"),
    ]
    rfe_cands = split_candidates(args.rfe_list) if args.rfe_list else [
        os.path.join("XGBoost_RFE", "RFE_推荐子集_特征清单.csv"),
        os.path.join("XGBoost_RFE", "RFE_recommended_features.csv"),
    ]

    # dataset paths
    X_full, y = load_Xy(base_dir, args.x_csv, args.x_ascii_csv, args.y_csv)

    # feature_groups.csv path
    fg_path = args.feature_groups if args.feature_groups else os.path.join(base_dir, "feature_groups.csv")

    # meta (optional)
    meta_path = args.meta_json if args.meta_json else os.path.join(base_dir, "dataset_meta.json")
    if os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            _ = meta.get("target_name", "")
        except Exception:
            pass

    print(f"🖋 Serif = {SERIF_MAIN} | CJK fallback = {CJK_FALLBACK or 'None (ASCII-sanitized labels)'}")
    print(f"📐 Data: X={X_full.shape}, y={len(y)}")
    print(f"📂 Output root: {out_base}")

    all_cols = list(X_full.columns)

    group_tables = {}  # mode -> group_contrib_df
    for mode in tqdm(["ALL", "PEARSON", "RFE"], desc="SHAP by mode"):
        feats, note = decide_features_for_mode(
            mode, all_cols, base_dir,
            pearson_cands=pearson_cands,
            rfe_cands=rfe_cands,
            scan_fallback=bool(args.scan_fallback),
            pearson_scan_dirs=pearson_scan_dirs,
            rfe_scan_dirs=rfe_scan_dirs
        )

        out_dir = os.path.join(out_base, mode)
        ensure_dir(out_dir)

        with open(os.path.join(out_dir, "feature_source.txt"), "w", encoding="utf-8") as f:
            f.write(note + "\n")
            f.write(f"n_features = {len(feats)}\n")

        df_imp, grp = run_one_mode(
            mode=mode,
            X_full=X_full,
            y=y,
            feats=feats,
            out_dir=out_dir,
            seed=args.seed,
            shap_sample=args.shap_sample,
            shap_topk=args.shap_topk,
            bsw_top=args.shap_beeswarm_top,
            dep_top=args.shap_depend_top,
            save_npz=bool(args.save_npz),
            do_beeswarm=bool(args.do_beeswarm),
            do_dependence=bool(args.do_dependence),
            fg_path=fg_path,
        )

        if grp is not None and not grp.empty:
            group_tables[mode] = grp[["group", "contribution_pct"]].copy()

    # Cross-mode group compare (if any)
    if group_tables:
        # union groups
        all_groups = sorted(set().union(*[set(df["group"].astype(str).tolist()) for df in group_tables.values()]))
        cmp = pd.DataFrame({"group": all_groups})
        for mode, df in group_tables.items():
            tmp = df.copy()
            tmp["group"] = tmp["group"].astype(str)
            cmp = cmp.merge(tmp.rename(columns={"contribution_pct": f"{mode}_pct"}), on="group", how="left")
        for c in [col for col in cmp.columns if col.endswith("_pct")]:
            cmp[c] = cmp[c].fillna(0.0)
        cmp.to_csv(os.path.join(out_base, "SHAP_group_compare_all_modes.csv"), index=False, encoding="utf-8-sig")

    print("\n✅ Done. Outputs →", out_base)


if __name__ == "__main__":
    main()
