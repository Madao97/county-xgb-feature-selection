# county-xgb-feature-selection
Reproducible XGBoost workflow for county-level economic modeling with LOYO validation, feature strategy comparison (ALL, Pearson-pruned, RFE), SHAP-based interpretation, and overfitting risk assessment.
<img width="1249" height="804" alt="fig2" src="https://github.com/user-attachments/assets/ff8c56cd-e9a9-4997-8c32-46cbfd15341c" />
# county-xgb-feature-selection

An end-to-end, reproducible pipeline for county-level modeling with **multi-source feature fusion** (remote sensing + socioeconomic/auxiliary data), **XGBoost**, **LOYO (Leave-One-Year-Out) validation**, and **SHAP interpretability**.  
Designed for limited-sample, multi-year panel datasets where temporal generalization and explainability matter.

---

##  What this repo does

- **Fuses multi-source data into a county-level tabular dataset** (prepared upstream), e.g.:
  - Remote sensing proxies (e.g., vegetation, land cover, night-time lights, climate/environment summaries)
  - Socioeconomic and infrastructure variables (e.g., POI, accessibility, public services, industry structure)
- Trains **XGBoost regressors** with **LOYO** cross-validation (realistic temporal generalization).
- Compares three feature strategies:
  - **ALL**: full feature set
  - **PEARSON**: correlation-pruned feature set
  - **RFE**: LOYO-guided recursive feature elimination (minimal “good-enough” subset)
- Produces **publication-ready** outputs:
  - OOF predictions, overall and year-wise metrics
  - Scatter-density plots and residual plots
  - Side-by-side comparisons across strategies
- Provides interpretability and robustness checks:
  - **SHAP** feature contribution (%), optional group contribution, beeswarm and dependence plots
  - **Stability check** using SHAP Top-20% simplified model to gauge overfitting risk

> Note: This repo focuses on modeling/selection/explainability. Data preprocessing and feature construction are assumed to be done before running these scripts.

---
