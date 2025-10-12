"""
Task 6 — MNI-vs-Population Comparison (2-file version)

WHAT THIS DOES
--------------
Reads:
  1) A CSV of ROI z-scores for ALL subjects (population)
  2) A CSV of ROI z-scores for the SINGLE MNI subject

Then it:
  - Aligns ROI features between the two files (intersection of ROI names)
  - Computes distances (Euclidean + Cosine) from the MNI subject to each population subject
  - Saves a ranked bar plot of distances and a 2D PCA scatter highlighting MNI
  - Writes a distances table (CSV) and the list of ROIs used

HOW TO USE
----------
1) Edit the two file paths below (population_csv, mni_csv). They are pre-filled with your provided files.
2) Edit optional settings (subject_col, standardize, metric, outdir) if needed.
3) Run:  python task6_mni_vs_population.py

ASSUMPTIONS / FLEXIBLE FORMATS
------------------------------
- WIDE format (preferred): one row per subject, ROI columns are numeric z-scores.
  * A subject ID column is recommended (e.g., "subject", "id", "SubjectID"). If not present, row index is used.
- LONG format (also supported): columns like ["subject","ROI","z"] or ["subject","roi","value"] etc.
  * The loader will try to pivot to wide format automatically.

PLOTS (PNG)
-----------
- distances_ranked.png : ranked bar plot of chosen metric
- pca_scatter.png      : PCA(2D) of population+MNI with MNI highlighted

OUTPUTS
-------
- distances.csv
- features_used.txt
- README.txt
"""

import os
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

# ============================
# 🔧 USER SETTINGS (edit here)
# ============================
# Your real files (pre-filled from your message)
population_csv = "/Users/harry/Desktop/ICM/Final Proect/git/ROI_z_scores.csv"  # ROI z-scores for ALL subjects
mni_csv        = "/Users/harry/Desktop/ICM/Final Proect/git/MNI Subject/z_scores.csv"  # ROI z-scores for MNI subject (single subject)

# If your population file has a subject ID column, put its name here; else set to None to auto-detect or use row index
subject_col: Optional[str] = None

# Distances + viz settings
standardize: bool = False   # Usually False for z-scores; set True if your features are raw and need scaling
metric: str = "euclidean"   # "euclidean" or "cosine" (for the ranked bar plot)
outdir: str = "/Users/harry/Desktop/ICM/Final Proect/git/task_6"
# ============================


# --------- Helpers for loading & shaping data ----------
def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def _guess_subject_col(df: pd.DataFrame) -> Optional[str]:
    """Heuristic: pick the first non-numeric column as subject ID if any; else None."""
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return non_numeric[0] if non_numeric else None


def _is_likely_long_format(df: pd.DataFrame) -> bool:
    """A rough check: long format typically has columns like subject/roi/value and fewer columns total."""
    cols = {c.lower() for c in df.columns}
    has_roi = any(k in cols for k in ["roi", "region", "roiname"])
    has_val = any(k in cols for k in ["z", "zscore", "value", "score"])
    has_sub = any(k in cols for k in ["subject", "id", "subject_id", "subjid"])
    return has_roi and has_val and has_sub and (df.shape[1] <= 5)


def _pivot_long_to_wide(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Convert long format (subject, ROI, value) to wide matrix: rows=subjects, cols=ROI.
    Tries flexible column name detection.
    """
    # Detect columns by lowercase name
    lower_map = {c.lower(): c for c in df.columns}
    # Subject
    subj_candidates = [k for k in ["subject", "subject_id", "id", "subjid"] if k in lower_map]
    if not subj_candidates:
        raise ValueError("Long format detected but could not find a subject ID column.")
    subj_col = lower_map[subj_candidates[0]]
    # ROI name
    roi_candidates = [k for k in ["roi", "region", "roiname"] if k in lower_map]
    if not roi_candidates:
        raise ValueError("Long format detected but could not find an ROI column.")
    roi_col = lower_map[roi_candidates[0]]
    # Value column (z-score)
    val_candidates = [k for k in ["z", "zscore", "value", "score"] if k in lower_map]
    if not val_candidates:
        # last resort – choose the first numeric column that is not ROI/subject
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("Long format detected but could not find a numeric value column.")
        val_col = numeric_cols[0]
    else:
        val_col = lower_map[val_candidates[0]]

    wide = df.pivot_table(index=subj_col, columns=roi_col, values=val_col, aggfunc="mean")
    wide = wide.sort_index()
    wide = wide.reset_index()
    # subject column remains as its original name
    feature_cols = [c for c in wide.columns if c != subj_col]
    return wide, subj_col, feature_cols


def _load_population_matrix(pop_csv: str, user_subject_col: Optional[str]) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Load population CSV and return (wide_df, subject_col, feature_cols).
    Supports both wide and long formats.
    """
    df = pd.read_csv(pop_csv)
    # LONG?
    if _is_likely_long_format(df):
        wide, subj_col, features = _pivot_long_to_wide(df)
        return wide, subj_col, features

    # WIDE: determine subject column
    subj_col = user_subject_col or _guess_subject_col(df)
    if subj_col is None:
        # No subject column — fabricate an index-based subject ID
        df = df.copy()
        df.insert(0, "subject", [f"S{i}" for i in range(len(df))])
        subj_col = "subject"

    # Feature columns: numeric and not the subject col
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in numeric_cols if c != subj_col]
    if not feature_cols:
        # If nothing numeric, try all columns except subject and convert to numeric where possible
        candidates = [c for c in df.columns if c != subj_col]
        try:
            df[candidates] = df[candidates].apply(pd.to_numeric, errors="raise")
            feature_cols = candidates
        except Exception as e:
            raise ValueError("Could not identify numeric ROI columns in population CSV.") from e

    return df, subj_col, feature_cols


def _load_mni_vector(mni_csv: str) -> pd.Series:
    """
    Load the MNI file and return a single-row vector of ROI z-scores as a pandas Series.
    Supports:
      - wide single-row CSV (numeric ROI columns)
      - long CSV with columns like [ROI, z]
    """
    df = pd.read_csv(mni_csv)

    # LONG vector?
    if _is_likely_long_format(df):
        wide, subj_col, feature_cols = _pivot_long_to_wide(df)
        if wide.shape[0] != 1:
            raise ValueError("MNI CSV (long) must contain exactly one subject.")
        row = wide.iloc[0]
        return row.drop(labels=[subj_col])

    # WIDE: expect exactly one row after ignoring any subject column
    subj_col_guess = _guess_subject_col(df)
    if df.shape[0] != 1:
        # If multiple rows, try to require 1 distinct subject
        if subj_col_guess and df[subj_col_guess].nunique() == 1:
            df = df.iloc[[0]].copy()
        else:
            raise ValueError("MNI CSV (wide) must contain exactly one row (one subject).")

    row = df.iloc[0]
    # Drop subject column if present
    if subj_col_guess is not None and subj_col_guess in row.index:
        row = row.drop(labels=[subj_col_guess])

    # Keep only numeric values (ROI z-scores)
    numeric = pd.to_numeric(row, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        raise ValueError("MNI CSV does not contain numeric ROI z-scores.")
    numeric.index.name = None
    return numeric


# --------- Distance + Visualization ----------
def compute_distances(
    pop_df: pd.DataFrame,
    subject_col: str,
    feature_cols: List[str],
    mni_series: pd.Series,
    standardize_features: bool
) -> pd.DataFrame:
    """
    Align ROIs between population and MNI, compute distances for each subject.
    Returns DataFrame with [subject, euclidean, cosine].
    """
    # Align ROI columns (intersection)
    pop_rois = set(feature_cols)
    mni_rois = set(mni_series.index.astype(str))
    common = sorted(list(pop_rois & mni_rois))
    if len(common) == 0:
        raise ValueError("No overlapping ROI names between population and MNI files.")

    X = pop_df[common].to_numpy(dtype=float)
    mni_vec = mni_series[common].to_numpy(dtype=float).reshape(1, -1)

    if standardize_features:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)
        mni_vec = scaler.transform(mni_vec)

    eu = euclidean_distances(X, mni_vec).reshape(-1)
    co = cosine_distances(X, mni_vec).reshape(-1)

    return pd.DataFrame({
        subject_col: pop_df[subject_col].astype(str).values,
        "euclidean": eu,
        "cosine": co
    }), common


def plot_ranked(dist_df: pd.DataFrame, subject_col: str, metric: str, out_png: str) -> None:
    data = dist_df.sort_values(metric, ascending=True).reset_index(drop=True)
    labels = data[subject_col].tolist()
    values = data[metric].to_numpy(dtype=float)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.xlabel("Subject")
    plt.ylabel(f"{metric.capitalize()} distance to MNI")
    plt.title(f"Ranked {metric.capitalize()} Distances to MNI Subject")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pca_with_mni(
    pop_df: pd.DataFrame,
    subject_col: str,
    roi_cols: List[str],
    mni_series: pd.Series,
    out_png: str,
    standardize_features: bool
) -> None:
    """
    PCA on population + MNI (appended), highlight MNI with a star.
    """
    # Build combined matrix
    X_pop = pop_df[roi_cols].to_numpy(dtype=float)
    x_mni = mni_series[roi_cols].to_numpy(dtype=float).reshape(1, -1)
    X_all = np.vstack([X_pop, x_mni])

    if standardize_features:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_all = scaler.fit_transform(X_all)

    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X_all)

    n_pop = X_pop.shape[0]
    mni_pt = X2[n_pop, :]

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:n_pop, 0], X2[:n_pop, 1], alpha=0.7, label="Population")
    plt.scatter(mni_pt[0], mni_pt[1], s=140, marker="*", label="MNI")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA: MNI vs Population (aligned ROIs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# --------- Main ----------
def main():
    out_path = _ensure_outdir(outdir)

    # Load population
    pop_df, subj_col_resolved, feature_cols = _load_population_matrix(population_csv, subject_col)

    # Load MNI vector
    mni_vec = _load_mni_vector(mni_csv)

    # Compute distances (and aligned ROI list actually used)
    dist_df, used_rois = compute_distances(
        pop_df=pop_df,
        subject_col=subj_col_resolved,
        feature_cols=feature_cols,
        mni_series=mni_vec,
        standardize_features=standardize
    )

    # Save outputs
    dist_csv = os.path.join(out_path, "distances.csv")
    dist_df.to_csv(dist_csv, index=False)

    with open(os.path.join(out_path, "features_used.txt"), "w", encoding="utf-8") as f:
        for r in used_rois:
            f.write(f"{r}\n")

    with open(os.path.join(out_path, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Task 6: MNI vs Population Comparison\n")
        f.write(f"- Population file: {population_csv}\n")
        f.write(f"- MNI file      : {mni_csv}\n")
        f.write(f"- Subject column: {subj_col_resolved}\n")
        f.write(f"- Standardize   : {standardize}\n")
        f.write(f"- Metric (plot) : {metric}\n")
        f.write(f"- #ROIs used    : {len(used_rois)}\n")

    # Visualizations
    ranked_png = os.path.join(out_path, "distances_ranked.png")
    chosen_metric = metric if metric in ("euclidean", "cosine") else "euclidean"
    plot_ranked(dist_df, subject_col=subj_col_resolved, metric=chosen_metric, out_png=ranked_png)

    pca_png = os.path.join(out_path, "pca_scatter.png")
    plot_pca_with_mni(
        pop_df=pop_df,
        subject_col=subj_col_resolved,
        roi_cols=used_rois,
        mni_series=mni_vec,
        out_png=pca_png,
        standardize_features=standardize
    )

    print("✅ Completed Task 6.")
    print(f"Outputs written to: {out_path}")
    print(f"- {os.path.basename(dist_csv)}")
    print(f"- distances_ranked.png")
    print(f"- pca_scatter.png")
    print(f"- features_used.txt")
    print("- README.txt")


if __name__ == "__main__":
    main()
