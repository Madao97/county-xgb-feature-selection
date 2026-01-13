# -*- coding: utf-8 -*-
"""
03_rfe_feature_selection.py

Purpose
-------
Recursive Feature Elimination (RFE) with Leave-One-Year-Out (LOYO) evaluation to identify a
"minimal yet sufficient" feature subset.

Workflow
--------
1) Load X.csv and y.csv (and optionally id.csv for the year column)
2) Determine initial feature set (optional external lists)
3) Iteratively:
   - Evaluate current feature set using LOYO CV (OOF predictions)
   - Fit a full model on all samples and rank features (XGBoost gain by default, SHAP optional)
   - Remove the weakest features by a fraction (STEP_FRAC) or at least STEP_MIN_FEATURES
4) Save the RFE path metrics and recommend an elbow subset based on tolerances:
   - R² >= best_R2 - TOL_R2
   - wMAPE(%) <= best_wMAPE + TOL_WMAPE
   Choose the smallest feature count satisfying both constraints.

Expected inputs
---------------
- X.csv: feature matrix (rows=samples, columns=features). May optionally include year_col.
- y.csv: target table with column y_col (default "target").
- id.csv (optional): sample identifiers containing year_col for LOYO splitting.

Optional inputs
---------------
- initial_features.csv: column "feature" listing candidate features to start from
- kept_features_csv: a kept list from pruning, with a single column containing features
  (e.g., "kept_feature" or similar; the script will auto-detect)

Usage
-----
python 03_rfe_feature_selection.py ^
  --x data/X.csv ^
  --y data/y.csv ^
  --out_dir outputs/rfe ^
  --id_csv data/id.csv ^
  --year_col year ^
  --y_col target ^
  --initial_features selected_features.csv ^
  --kept_features_csv pearson_pruning_kept.csv ^
  --step_frac 0.10 ^
  --min_features_keep 15

Outputs
-------
- rfe_path_metrics.csv
- rfe_removed_per_step.csv
- rfe_recommended_features.csv
- rfe_curve_r2.png
- rfe_curve_wmape.png
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

from xgboost import XGBRegressor


# =========================
# Metrics
# =========================
EPS = 1e-8

def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def wmape(y_true, y_pred, eps=EPS) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))

def smape(y_true, y_pred, eps=EPS) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)))

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {"r2": r2_score(y_true, y_pred), "wmape": wmape(y_true, y_pred), "smape": smape(y_true, y_pred)}


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
# Feature list helpers
# =========================
def load_feature_list_csv(path: Optional[str], X_cols: List[str]) -> Optional[List[str]]:
    if not path or not os.path.isfile(path):
        return None
    df = read_csv(path)
    if "feature" not in df.columns:
        return None
    feats = [f for f in df["feature"].astype(str).tolist() if f in X_cols]
    return feats if len(feats) >= 1 else None

def load_kept_list_csv(path: Optional[str], X_cols: List[str]) -> Optional[List[str]]:
    """
    Accept a kept list CSV with an arbitrary single column name, or common names like:
    kept_feature, feature, kept.
    """
    if not path or not os.path.isfile(path):
        return None
    df = read_csv(path)
    # try common column names first
    for cname in ["kept_feature", "feature", "kept"]:
        if cname in df.columns:
            feats = [f for f in df[cname].dropna().astype(str).tolist() if f in X_cols]
            return feats if len(feats) >= 1 else None
    # fallback: use the first column
    first_col = df.columns[0]
    feats = [f for f in df[first_col].dropna().astype(str).tolist() if f in X_cols]
    return feats if len(feats) >= 1 else None


# =========================
# Modeling
# =========================
def impute_median_train_valid(Xtr: pd.DataFrame, Xva: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    med = Xtr.median(axis=0, skipna=True)
    Xtr2 = Xtr.replace([np.inf, -np.inf], np.nan).fillna(med)
    Xva2 = Xva.replace([np.inf, -np.inf], np.nan).fillna(med)
    return Xtr2, Xva2

def make_model(seed: int, n_estimators: int, learning_rate: float, max_depth: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=-1,
    )

def loyo_oof_eval(
    X: pd.DataFrame,
    y: np.ndarray,
    years: np.ndarray,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    uniq_years = sorted(pd.unique(years))
    oof = np.zeros_like(y, dtype=float)

    for yv in uniq_years:
        tr_idx = np.where(years != yv)[0]
        va_idx = np.where(years == yv)[0]

        Xtr, ytr = X.iloc[tr_idx, :], y[tr_idx]
        Xva, yva = X.iloc[va_idx, :], y[va_idx]

        Xtr2, Xva2 = impute_median_train_valid(Xtr, Xva)
        model = make_model(seed, n_estimators, learning_rate, max_depth)
        model.fit(Xtr2, ytr, verbose=False)
        oof[va_idx] = model.predict(Xva2)

    return compute_metrics(y, oof), oof


# =========================
# Feature ranking
# =========================
def rank_features_by_gain(model: XGBRegressor, X_used: pd.DataFrame) -> pd.Series:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    # XGBoost may return "f0,f1,..." in some versions; map back if needed
    raw = pd.Series(gain, dtype=float)
    # ensure we have all columns
    if raw.index.str.startswith("f").all():
        # Map f0.. to column names by position
        mapping = {f"f{i}": col for i, col in enumerate(X_used.columns)}
        raw.index = [mapping.get(k, k) for k in raw.index]
    s = pd.Series({c: float(raw.get(c, 0.0)) for c in X_used.columns}, name="importance")
    return s.sort_values(ascending=False)

def rank_features_by_shap(model: XGBRegressor, X_used: pd.DataFrame, seed: int, shap_sample: int) -> pd.Series:
    import shap  # optional dependency
    rng = np.random.RandomState(seed)
    ns = min(int(shap_sample), X_used.shape[0])
    idx = rng.choice(X_used.index.values, size=ns, replace=False)
    Xs = X_used.loc[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)
    imp = pd.Series(np.abs(shap_values).mean(axis=0), index=X_used.columns, name="importance")
    return imp.sort_values(ascending=False)

def feature_ranking(
    model: XGBRegressor,
    X_used: pd.DataFrame,
    use_shap: bool,
    seed: int,
    shap_sample: int,
) -> pd.Series:
    if use_shap:
        try:
            return rank_features_by_shap(model, X_used, seed=seed, shap_sample=shap_sample)
        except Exception as e:
            print(f"SHAP ranking failed ({repr(e)}). Falling back to gain ranking.")
    return rank_features_by_gain(model, X_used)


# =========================
# CLI
# =========================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RFE with LOYO evaluation for XGBoost regression.")
    p.add_argument("--x", required=True, help="Path to X.csv")
    p.add_argument("--y", required=True, help="Path to y.csv")
    p.add_argument("--out_dir", required=True, help="Output directory")

    p.add_argument("--y_col", default="target", help="Target column name in y.csv")
    p.add_argument("--id_csv", default=None, help="Optional id.csv containing year column")
    p.add_argument("--year_col", default="year", help="Year column name for LOYO splitting (in id.csv or X.csv)")

    p.add_argument("--initial_features", default=None, help="Optional CSV with column 'feature' (initial candidates)")
    p.add_argument("--kept_features_csv", default=None, help="Optional kept list CSV from pruning")

    p.add_argument("--use_shap_for_rank", action="store_true", help="Use SHAP mean(|SHAP|) for ranking (slower)")
    p.add_argument("--shap_sample", type=int, default=3000, help="Sample size for SHAP ranking if enabled")

    p.add_argument("--step_frac", type=float, default=0.10, help="Fraction of features removed per step")
    p.add_argument("--step_min_features", type=int, default=1, help="Minimum number of features removed per step")
    p.add_argument("--min_features_keep", type=int, default=15, help="Stop when reaching this many features")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_estimators", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--max_depth", type=int, default=6)

    p.add_argument("--tol_r2", type=float, default=0.005, help="Tolerance for R² relative to best")
    p.add_argument("--tol_wmape", type=float, default=0.5, help="Tolerance for wMAPE (%) relative to best")

    return p


# =========================
# Main
# =========================
def main() -> None:
    args = build_arg_parser().parse_args()
    t0 = time.time()

    out_dir = ensure_dir(args.out_dir)

    # Load
    X_raw = read_csv(args.x)
    y_df = read_csv(args.y)
    if args.y_col not in y_df.columns:
        raise KeyError(f"Target column '{args.y_col}' not found in y.csv.")
    y = y_df[args.y_col].values

    # Year source: id.csv preferred
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

    # Build feature table (exclude year if present)
    X_feat = X_raw.copy()
    if args.year_col in X_feat.columns:
        X_feat = X_feat.drop(columns=[args.year_col])

    X_feat = coerce_numeric_df(X_feat)
    X_feat = drop_constant_columns(X_feat)

    if X_feat.shape[0] != len(y) or len(y) != len(years):
        raise ValueError("Row mismatch among X, y, and year column. Ensure consistent row order and counts.")

    # Initial feature set (priority: initial_features > kept_features_csv > all)
    X_cols = list(X_feat.columns)
    feats = load_feature_list_csv(args.initial_features, X_cols)
    if feats is None:
        feats = load_kept_list_csv(args.kept_features_csv, X_cols)
    if feats is None:
        feats = X_cols

    # Filter invalid rows (finite y and features)
    Xnum = X_feat[feats].copy()
    Xnum = Xnum.replace([np.inf, -np.inf], np.nan)
    mask = np.isfinite(y)
    mask = mask & np.isfinite(Xnum.apply(pd.to_numeric, errors="coerce")).all(axis=1).values

    Xnum = Xnum.loc[mask].reset_index(drop=True)
    y2 = y[mask]
    years2 = np.asarray(years)[mask]

    # RFE loop
    rows = []
    removed_records = []
    step_id = 0

    Xcurr = Xnum.copy()

    while True:
        n_feat = Xcurr.shape[1]
        if n_feat < max(int(args.min_features_keep), 2):
            break

        # 1) LOYO evaluation
        met, _oof = loyo_oof_eval(
            Xcurr, y2, years2,
            seed=args.seed,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
        )
        met.update({"step": step_id, "n_features": n_feat})
        rows.append(met)
        print(
            f"[RFE] step={step_id:02d}, n={n_feat:4d}, R²={met['r2']:.4f}, "
            f"wMAPE={met['wmape']*100:.2f}%, sMAPE={met['smape']*100:.2f}%"
        )

        # 2) Fit full model for ranking
        Xfit = Xcurr.replace([np.inf, -np.inf], np.nan)
        Xfit = Xfit.fillna(Xfit.median(axis=0, skipna=True))

        model = make_model(args.seed, args.n_estimators, args.learning_rate, args.max_depth)
        model.fit(Xfit, y2, verbose=False)

        imp = feature_ranking(
            model=model,
            X_used=Xfit,
            use_shap=bool(args.use_shap_for_rank),
            seed=args.seed,
            shap_sample=int(args.shap_sample),
        )

        # 3) Determine removal set
        k = max(int(args.step_min_features), int(np.ceil(float(args.step_frac) * n_feat)))
        k = min(k, n_feat - int(args.min_features_keep))  # do not go below lower bound
        if k <= 0:
            break

        drop_list = imp.sort_values(ascending=True).index[:k].tolist()
        keep_list = [c for c in Xcurr.columns if c not in drop_list]

        removed_records.append(
            {"step": step_id, "n_features": n_feat, "removed_count": len(drop_list), "removed_features": "|".join(drop_list)}
        )

        Xcurr = Xcurr[keep_list].copy()
        step_id += 1

    # Final evaluation for the last subset
    if Xcurr.shape[1] >= int(args.min_features_keep):
        met, _oof = loyo_oof_eval(
            Xcurr, y2, years2,
            seed=args.seed,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
        )
        met.update({"step": step_id, "n_features": Xcurr.shape[1]})
        rows.append(met)

    path_df = pd.DataFrame(rows).sort_values("n_features", ascending=False).reset_index(drop=True)
    path_df["wmape_pct"] = path_df["wmape"] * 100.0
    path_df["smape_pct"] = path_df["smape"] * 100.0
    path_df.to_csv(os.path.join(out_dir, "rfe_path_metrics.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame(removed_records).to_csv(os.path.join(out_dir, "rfe_removed_per_step.csv"),
                                         index=False, encoding="utf-8-sig")

    # Recommend elbow subset: smallest n satisfying tolerance constraints
    best_r2 = float(path_df["r2"].max())
    best_wmape = float(path_df["wmape_pct"].min())

    cand = path_df[
        (path_df["r2"] >= best_r2 - float(args.tol_r2)) &
        (path_df["wmape_pct"] <= best_wmape + float(args.tol_wmape))
    ].sort_values("n_features", ascending=True)

    if len(cand) > 0:
        n_star = int(cand.iloc[0]["n_features"])
    else:
        tmp = path_df.sort_values(["r2", "n_features"], ascending=[False, True])
        n_star = int(tmp.iloc[0]["n_features"])

    # Reconstruct recommended feature subset by replaying RFE from the initial feature set
    # (to make the output deterministic and self-contained)
    feats0 = feats[:]  # initial candidates
    Xtmp = Xnum[feats0].copy()

    while Xtmp.shape[1] > n_star:
        Xfit = Xtmp.replace([np.inf, -np.inf], np.nan)
        Xfit = Xfit.fillna(Xfit.median(axis=0, skipna=True))

        model = make_model(args.seed, args.n_estimators, args.learning_rate, args.max_depth)
        model.fit(Xfit, y2, verbose=False)

        imp = feature_ranking(
            model=model,
            X_used=Xfit,
            use_shap=bool(args.use_shap_for_rank),
            seed=args.seed,
            shap_sample=int(args.shap_sample),
        )

        k = max(int(args.step_min_features), int(np.ceil(float(args.step_frac) * Xtmp.shape[1])))
        k = min(k, Xtmp.shape[1] - n_star)
        drop_list = imp.sort_values(ascending=True).index[:k].tolist()
        keep_list = [c for c in Xtmp.columns if c not in drop_list]
        Xtmp = Xtmp[keep_list].copy()

    pd.DataFrame({"feature": list(Xtmp.columns)}).to_csv(
        os.path.join(out_dir, "rfe_recommended_features.csv"), index=False, encoding="utf-8-sig"
    )

    # Curves
    plt.figure(figsize=(10, 5), facecolor="white")
    plt.plot(path_df["n_features"], path_df["r2"], marker="o")
    plt.axvline(n_star, linestyle="--", linewidth=1.2)
    plt.gca().invert_xaxis()
    plt.xlabel("Number of features")
    plt.ylabel("R²")
    plt.title("RFE performance curve (R²)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rfe_curve_r2.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5), facecolor="white")
    plt.plot(path_df["n_features"], path_df["wmape_pct"], marker="o")
    plt.axvline(n_star, linestyle="--", linewidth=1.2)
    plt.gca().invert_xaxis()
    plt.xlabel("Number of features")
    plt.ylabel("wMAPE (%)")
    plt.title("RFE performance curve (wMAPE)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rfe_curve_wmape.png"), dpi=300)
    plt.close()

    print("\nRFE completed.")
    print("Output directory:", out_dir)
    print("Key outputs:")
    print("  - rfe_path_metrics.csv")
    print("  - rfe_removed_per_step.csv")
    print("  - rfe_recommended_features.csv")
    print("  - rfe_curve_r2.png / rfe_curve_wmape.png")
    print(f"Recommended feature count: n* = {n_star}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
