"""
Task 6 — MNI vs Population (PCA colored by demographics)

WHAT'S NEW
----------
In addition to the ranked distance plot and the base PCA scatter,
this version generates THREE extra PCA maps that color the POPULATION
by each demographic field (if present) while still highlighting the MNI:
  - Diagnose  (categorical)
  - age       (numeric, continuous)
  - gender    (categorical)

FILES (edit paths below)
------------------------
- population_csv: ROI z-scores for ALL subjects (wide or long format supported)
- mni_csv       : ROI z-scores for SINGLE MNI subject (wide one-row or long format)

OUTPUTS
-------
- distances.csv
- distances_ranked.png
- pca_scatter.png                 (population gray + MNI star)
- pca_scatter_by_Diagnose.png     (if 'Diagnose' found)
- pca_scatter_by_age.png          (if 'age' found)
- pca_scatter_by_gender.png       (if 'gender' found)
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
population_csv = "/Users/harry/Desktop/ICM/Final Proect/git/ROI_z_scores.csv"  # ROI z-scores for ALL subjects
mni_csv        = "/Users/harry/Desktop/ICM/Final Proect/git/MNI Subject/z_scores.csv"  # ROI z-scores for MNI subject (single subject)

# If your population file has a subject ID column name, put it here (else None to auto-guess or use row index)
subject_col: Optional[str] = None

# Distances + viz settings
standardize: bool = False   # For z-scores, usually False; set True if features need scaling
metric: str = "euclidean"   # "euclidean" or "cosine" for ranked bar plot
outdir: str = "/Users/harry/Desktop/ICM/Final Proect/git/task_6"
# ============================


# --------- Helpers for loading & shaping data ----------
def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)

def _guess_subject_col(df: pd.DataFrame) -> Optional[str]:
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return non_numeric[0] if non_numeric else None

def _is_likely_long_format(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    has_roi = any(k in cols for k in ["roi", "region", "roiname"])
    has_val = any(k in cols for k in ["z", "zscore", "value", "score"])
    has_sub = any(k in cols for k in ["subject", "id", "subject_id", "subjid"])
    return has_roi and has_val and has_sub and (df.shape[1] <= 5)

def _pivot_long_to_wide(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str]]:
    lower_map = {c.lower(): c for c in df.columns}
    subj_col = next((lower_map[k] for k in ["subject","subject_id","id","subjid"] if k in lower_map), None)
    if subj_col is None:
        raise ValueError("Long format detected but could not find a subject ID column.")
    roi_col  = next((lower_map[k] for k in ["roi","region","roiname"] if k in lower_map), None)
    if roi_col is None:
        raise ValueError("Long format detected but could not find an ROI column.")
    val_col  = next((lower_map[k] for k in ["z","zscore","value","score"] if k in lower_map), None)
    if val_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("Long format detected but could not find a numeric value column.")
        val_col = numeric_cols[0]
    wide = df.pivot_table(index=subj_col, columns=roi_col, values=val_col, aggfunc="mean")
    wide = wide.sort_index().reset_index()
    feature_cols = [c for c in wide.columns if c != subj_col]
    return wide, subj_col, feature_cols

def _load_population_matrix(pop_csv: str, user_subject_col: Optional[str]) -> Tuple[pd.DataFrame, str, List[str]]:
    df = pd.read_csv(pop_csv)
    if _is_likely_long_format(df):
        return _pivot_long_to_wide(df)

    subj_col = user_subject_col or _guess_subject_col(df)
    if subj_col is None:
        df = df.copy()
        df.insert(0, "subject", [f"S{i}" for i in range(len(df))])
        subj_col = "subject"

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in numeric_cols if c != subj_col]
    if not feature_cols:
        candidates = [c for c in df.columns if c != subj_col]
        df[candidates] = df[candidates].apply(pd.to_numeric, errors="raise")
        feature_cols = candidates
    return df, subj_col, feature_cols

def _load_mni_vector(mni_csv: str) -> pd.Series:
    df = pd.read_csv(mni_csv)
    if _is_likely_long_format(df):
        wide, subj_col, feature_cols = _pivot_long_to_wide(df)
        if wide.shape[0] != 1:
            raise ValueError("MNI CSV (long) must contain exactly one subject.")
        row = wide.iloc[0]
        return row.drop(labels=[subj_col])

    subj_guess = _guess_subject_col(df)
    if df.shape[0] != 1:
        if subj_guess and df[subj_guess].nunique() == 1:
            df = df.iloc[[0]].copy()
        else:
            raise ValueError("MNI CSV (wide) must contain exactly one row (one subject).")
    row = df.iloc[0]
    if subj_guess is not None and subj_guess in row.index:
        row = row.drop(labels=[subj_guess])
    numeric = pd.to_numeric(row, errors="coerce").dropna()
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
) -> Tuple[pd.DataFrame, List[str]]:
    pop_rois = set(feature_cols)
    mni_rois = set(map(str, mni_series.index))
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

def _compute_pca_coordinates(pop_df: pd.DataFrame, roi_cols: List[str], mni_series: pd.Series, standardize_features: bool):
    X_pop = pop_df[roi_cols].to_numpy(dtype=float)
    x_mni = mni_series[roi_cols].to_numpy(dtype=float).reshape(1, -1)
    X_all = np.vstack([X_pop, x_mni])
    if standardize_features:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_all = scaler.fit_transform(X_all)
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X_all)
    n_pop = X_pop.shape[0]
    return X2[:n_pop, :], X2[n_pop, :]

def plot_pca_base(pop_df: pd.DataFrame, roi_cols: List[str], mni_series: pd.Series, out_png: str, standardize_features: bool):
    X2_pop, mni_pt = _compute_pca_coordinates(pop_df, roi_cols, mni_series, standardize_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(X2_pop[:, 0], X2_pop[:, 1], alpha=0.7, label="Population")
    plt.scatter(mni_pt[0], mni_pt[1], s=140, marker="*", label="MNI")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA: MNI vs Population (aligned ROIs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_pca_by_category(
    pop_df: pd.DataFrame,
    roi_cols: List[str],
    mni_series: pd.Series,
    category_col: str,
    out_png: str,
    standardize_features: bool
) -> None:
    """
    PCA colored by a categorical column (e.g., gender, diagnosis).
    - Each category is plotted as a separate layer for a clean legend.
    - Missing values are treated as a distinct "Missing" category.
    - MNI point is highlighted with a star.
    """
    if category_col not in pop_df.columns:
        return

    # Prepare categories (treat missing as a category)
    cats = pop_df[category_col].astype("object").fillna("Missing")
    cats = pd.Series(cats, dtype="category")
    if cats.cat.categories.size == 0:
        return

    # Compute PCA coordinates (population + MNI)
    X2_pop, mni_pt = _compute_pca_coordinates(
        pop_df=pop_df, roi_cols=roi_cols, mni_series=mni_series, standardize_features=standardize_features
    )

    # Plot each category as a separate scatter for a readable legend
    plt.figure(figsize=(8, 6))
    handles = []
    labels = []
    for cat in cats.cat.categories:
        idx = np.where(cats.values == cat)[0]
        if idx.size == 0:
            continue
        h = plt.scatter(X2_pop[idx, 0], X2_pop[idx, 1], alpha=0.85)
        handles.append(h)
        labels.append(str(cat))

    # Highlight MNI
    plt.scatter(mni_pt[0], mni_pt[1], s=160, marker="*", label="MNI")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA colored by {category_col}")
    if handles:
        plt.legend(handles + [plt.Line2D([], [], linestyle="none", marker="*", markersize=10)],
                   labels + ["MNI"], title=category_col, loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pca_by_numeric(
    pop_df: pd.DataFrame,
    roi_cols: List[str],
    mni_series: pd.Series,
    numeric_col: str,
    out_png: str,
    standardize_features: bool
) -> None:
    """
    PCA colored by a numeric column (e.g., age) with a smooth gradient.
    - Values are coerced to numeric for coloring.
    - Colorbar shows integer ticks spanning the observed range.
    - Points with NaN in the numeric column are still plotted (without contributing to the color scale).
    - MNI point is highlighted with a star.
    """
    if numeric_col not in pop_df.columns:
        return

    vals = pd.to_numeric(pop_df[numeric_col], errors="coerce")
    if vals.notna().sum() == 0:
        # Nothing numeric to color by
        return

    # Integer-tick display (do not modify actual values)
    int_vals = vals.round().astype("Int64")

    # Compute PCA coordinates (population + MNI)
    X2_pop, mni_pt = _compute_pca_coordinates(
        pop_df=pop_df, roi_cols=roi_cols, mni_series=mni_series, standardize_features=standardize_features
    )

    # Scatter with continuous colormap; NaN values get default facecolor
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2_pop[:, 0], X2_pop[:, 1], c=vals, alpha=0.9, cmap='viridis')

    # Colorbar with integer ticks
    cbar = plt.colorbar(sc)
    cbar.set_label(numeric_col)

    if int_vals.notna().any():
        vmin = int(int_vals.min())
        vmax = int(int_vals.max())
        if vmin == vmax:
            ticks = [vmin]
        else:
            # Choose up to ~6 nicely spaced integer ticks
            span = vmax - vmin
            step = max(1, span // 5)
            ticks = list(range(vmin, vmax + 1, step))
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])

    # Highlight MNI
    plt.scatter(mni_pt[0], mni_pt[1], s=160, marker="*", c='red', edgecolors='black', linewidths=1.5, label="MNI", zorder=5)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA colored by {numeric_col} (continuous)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# --------- Main ----------
def main():
    out_path = _ensure_outdir(outdir)

    # Load population + MNI
    pop_df, subj_col_resolved, feature_cols = _load_population_matrix(population_csv, subject_col)
    mni_vec = _load_mni_vector(mni_csv)

    # Distances + ROI alignment actually used
    dist_df, used_rois = compute_distances(
        pop_df=pop_df,
        subject_col=subj_col_resolved,
        feature_cols=feature_cols,
        mni_series=mni_vec,
        standardize_features=standardize
    )

    # Save table + feature list
    dist_csv = os.path.join(out_path, "distances.csv")
    dist_df.to_csv(dist_csv, index=False)
    with open(os.path.join(out_path, "features_used.txt"), "w", encoding="utf-8") as f:
        for r in used_rois:
            f.write(f"{r}\n")

    # README
    with open(os.path.join(out_path, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Task 6: MNI vs Population Comparison\n")
        f.write(f"- Population file: {population_csv}\n")
        f.write(f"- MNI file      : {mni_csv}\n")
        f.write(f"- Subject column: {subj_col_resolved}\n")
        f.write(f"- Standardize   : {standardize}\n")
        f.write(f"- Metric (plot) : {metric}\n")
        f.write(f"- #ROIs used    : {len(used_rois)}\n")

    # Visualizations (ranked + base PCA)
    plot_ranked(dist_df, subject_col=subj_col_resolved,
                metric=(metric if metric in ("euclidean", "cosine") else "euclidean"),
                out_png=os.path.join(out_path, "distances_ranked.png"))
    plot_pca_base(pop_df, used_rois, mni_vec,
                  out_png=os.path.join(out_path, "pca_scatter.png"),
                  standardize_features=standardize)

    # Demographic-aware PCA maps (case-insensitive matching to 'Diagnose', 'age', 'gender')
    # Build a case-insensitive mapping
    lower_map = {c.lower(): c for c in pop_df.columns}
    
    # Diagnosis (categorical)
    for key in ["diagnose", "diagnosis", "dx"]:
        if key in lower_map:
            plot_pca_by_category(pop_df, used_rois, mni_vec,
                                 category_col=lower_map[key],
                                 out_png=os.path.join(out_path, "pca_scatter_by_Diagnose.png"),
                                 standardize_features=standardize)
            break
    
    # Gender (categorical) - FIXED: use lowercase keys
    for key in ["gender", "sex"]:
        if key in lower_map:
            plot_pca_by_category(pop_df, used_rois, mni_vec,
                                 category_col=lower_map[key],
                                 out_png=os.path.join(out_path, "pca_scatter_by_gender.png"),
                                 standardize_features=standardize)
            break
    
    # Age (numeric) - FIXED: use lowercase keys
    for key in ["age", "age_years"]:
        if key in lower_map:
            plot_pca_by_numeric(pop_df, used_rois, mni_vec,
                                numeric_col=lower_map[key],
                                out_png=os.path.join(out_path, "pca_scatter_by_age.png"),
                                standardize_features=standardize)
            break

    print("✅ Completed Task 6 with demographic-colored PCA maps.")
    print(f"Outputs written to: {out_path}")

if __name__ == "__main__":
    main()