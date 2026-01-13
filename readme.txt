County-Level Economic Modeling with XGBoost and SHAP

This repository provides a reproducible, end-to-end machine learning workflow for county-level economic assessment using XGBoost.
The pipeline is designed for limited-sample, multi-year panel data, with a focus on robust generalization, feature strategy comparison, and interpretable inference.

1. Key Features

Leave-One-Year-Out (LOYO) cross-validation for realistic temporal generalization

Multiple feature strategies:

ALL (full feature set)

Pearson correlation–pruned

RFE-recommended minimal subset

Recursive Feature Elimination (RFE) guided by LOYO performance

SHAP-based interpretability:

Feature-level contribution

Group-level contribution

Dependence and beeswarm plots

Overfitting risk assessment via SHAP Top-20% feature simplification

Publication-ready outputs:

OOF predictions

Overall and year-wise metrics

High-resolution figures (scatter density, residuals, comparisons)

2. Data Interface

The pipeline assumes a tabular county-level dataset with the following required files:

Required Files
File	Description
X.csv	Feature matrix (numeric or convertible to numeric)
X_with_id.csv	Same rows as X.csv, including ID columns and a year column
y.csv	Target variable with column name target
dataset_meta.json	(Optional) Target name and unit for figure labels
Required Columns

A year column named 年份 or year

Target column named target

Optional Files
File	Purpose
selected_features_pearson_global.csv	Candidate feature list for Pearson mode
RFE_推荐子集_特征清单.csv	Recommended feature subset from RFE
feature_groups.csv	Feature-to-group mapping for grouped SHAP analysis
3. Environment Setup
Python Version

Python ≥ 3.9 recommended

Dependencies
pip install numpy pandas scipy matplotlib tqdm xgboost shap


(Optional) For better Chinese label rendering in figures, install a CJK font such as:

Microsoft YaHei

SimHei

Noto Sans CJK

4. Pipeline Overview

The workflow consists of seven core scripts, executed sequentially or independently.

(01) Dataset Construction

Builds training tables:

X.csv

X_with_id.csv

y.csv

dataset_meta.json

(02) Correlation-Based Screening (Optional)

Generates a global candidate feature list to stabilize downstream pruning.

(03) RFE with LOYO Validation
python src/03_rfe_feature_selection.py


Performs recursive feature elimination

Evaluates each step using LOYO cross-validation

Automatically selects a minimal yet sufficient feature subset

Key outputs

RFE performance path (R², wMAPE)

Recommended feature subset

Performance curves

(04) Single-Strategy Training and Interpretation
python src/04_train_xgboost_rfe.py


Trains XGBoost models under one feature strategy

Produces OOF predictions, metrics, and SHAP interpretation

(05) Feature Strategy Comparison
python src/05_compare_feature_strategies.py


Compares ALL / Pearson / RFE strategies with:

Side-by-side overall scatter plots

Year-wise comparison figures

Metric comparison tables

(06) SHAP-Based Stability Check
python src/06_shap_stability_check.py


Retains only SHAP Top-20% features

Retrains simplified models

Compares full vs simplified performance to assess overfitting risk

(07) SHAP Group Analysis (CJK-safe)
python src/07_shap_group_analysis.py


Full SHAP analysis for each feature strategy

Beeswarm, dependence, and grouped contribution plots

Automatic fallback for environments without CJK fonts

5. Cross-Validation and Reproducibility

Cross-validation: Leave-One-Year-Out (LOYO)

Missing values: Median imputation fitted on training folds only

Random seed: Fixed to ensure reproducible training and SHAP sampling

6. Output Structure

Typical outputs include:

OOF prediction tables (CSV)

Overall and year-wise metrics

Publication-ready figures (300 DPI)

Trained XGBoost models (JSON)

SHAP contribution summaries

Large outputs are recommended to be excluded from version control via .gitignore.

7. Intended Use

This repository is designed for:

County-level or regional economic modeling

Socioeconomic indicator estimation with limited samples

Methodological studies on feature selection and model interpretability

Reproducible research supporting journal publication

8. License

Choose an appropriate open-source license (e.g., MIT or Apache-2.0) depending on your data-sharing policy.