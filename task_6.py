import os
import sys
import argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import matplotlib.pyplot as plt


def _ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return os.path.abspath(outdir)


def _infer_columns(
    df: pd.DataFrame,
    subject_col: str,
    is_mni_col: str = "is_mni"
) -> Tuple[str, Optional[str], List[str]]:
    """
    Determine subject and MNI columns and which columns are numeric features.
    """
    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found in CSV. Available: {list(df.columns)}")
    is_mni_present = is_mni_col in df.columns
    # Feature columns: numeric columns excluding id/flags
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != is_mni_col]
    if not feature_cols:
        raise ValueError("No numeric feature columns found. Ensure your barcode/features are numeric.")
    return subject_col, (is_mni_col if is_mni_present else None), feature_cols


def _get_mni_index(
    df: pd.DataFrame,
    subject_col: str,
    feature_cols: List[str],
    mni_id: Optional[str],
    is_mni_col: Optional[str]
) -> int:
    """
    Identify the row index of the MNI subject.
    Priority:
      1) mni_id argument if provided.
      2) is_mni boolean column with exactly one True.
      3) subject value equals one of common defaults: "MNI" (case-insensitive exact match).
    """
    if mni_id is not None:
        hits = df.index[df[subject_col].astype(str) == str(mni_id)].tolist()
        if len(hits) == 0:
            raise ValueError(f"Provided --mni-id '{mni_id}' not found in column '{subject_col}'.")
        if len(hits) > 1:
            raise ValueError(f"Provided --mni-id '{mni_id}' matches multiple rows; subject IDs must be unique.")
        return hits[0]

    if is_mni_col is not None:
        true_rows = df.index[df[is_mni_col].astype(bool)].tolist()
        if len(true_rows) == 1:
            return true_rows[0]
        elif len(true_rows) > 1:
            raise ValueError(f"Column '{is_mni_col}' has multiple True rows; expected exactly one.")

    subj_vals = df[subject_col].astype(str)
    hits = df.index[subj_vals.str.lower() == "mni"].tolist()
    if len(hits) == 1:
        return hits[0]

    raise ValueError(
        "Could not determine MNI subject. Provide --mni-id or include a single True in 'is_mni' column or a subject named 'MNI'."
    )


def compute_distances_to_mni(
    df: pd.DataFrame,
    subject_col: str,
    feature_cols: List[str],
    mni_idx: int,
    standardize: bool = True
) -> pd.DataFrame:
    """
    Compute Euclidean and Cosine distances between the MNI subject and all other subjects.
    """
    X = df[feature_cols].to_numpy(dtype=float)

    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)

    mni_vec = X[mni_idx, :].reshape(1, -1)
    eu = euclidean_distances(X, mni_vec).reshape(-1)
    co = cosine_distances(X, mni_vec).reshape(-1)

    out = pd.DataFrame({
        subject_col: df[subject_col].astype(str).values,
        "euclidean": eu,
        "cosine": co,
        "is_mni": False
    })
    out.loc[mni_idx, "is_mni"] = True
    return out


def plot_ranked_distances(
    dist_df: pd.DataFrame,
    subject_col: str,
    metric: str,
    out_png: str
) -> None:
    """
    Save a ranked bar plot of distances for the chosen metric.
    """
    if metric not in ("euclidean", "cosine"):
        raise ValueError("metric must be 'euclidean' or 'cosine'")

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


def plot_pca_scatter(
    df: pd.DataFrame,
    subject_col: str,
    feature_cols: List[str],
    mni_idx: int,
    out_png: str,
    standardize: bool = True
) -> None:
    """
    Save a 2D PCA scatter with MNI subject highlighted.
    """
    X = df[feature_cols].to_numpy(dtype=float)

    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1], alpha=0.7, label="Population")
    plt.scatter(X2[mni_idx, 0], X2[mni_idx, 1], s=120, marker="*", label="MNI")
    mni_label = str(df.iloc[mni_idx][subject_col])
    plt.annotate(mni_label, (X2[mni_idx, 0], X2[mni_idx, 1]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA: MNI vs Population")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_features_used(feature_cols: List[str], out_txt: str) -> None:
    with open(out_txt, "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(f"{c}\n")


def run(
    csv_path: str,
    subject_col: str = "subject",
    mni_id: Optional[str] = None,
    outdir: str = "./task6_results",
    standardize: bool = True,
    metric: str = "euclidean"
) -> str:
    outdir = _ensure_outdir(outdir)

    df = pd.read_csv(csv_path)
    subject_col, is_mni_col, feature_cols = _infer_columns(df, subject_col=subject_col)

    mni_idx = _get_mni_index(df, subject_col, feature_cols, mni_id, is_mni_col)

    dist_df = compute_distances_to_mni(
        df=df,
        subject_col=subject_col,
        feature_cols=feature_cols,
        mni_idx=mni_idx,
        standardize=standardize
    )

    distances_csv = os.path.join(outdir, "distances.csv")
    dist_df.to_csv(distances_csv, index=False)
    save_features_used(feature_cols, os.path.join(outdir, "features_used.txt"))

    ranked_png = os.path.join(outdir, "distances_ranked.png")
    plot_ranked_distances(dist_df, subject_col=subject_col, metric=metric, out_png=ranked_png)

    pca_png = os.path.join(outdir, "pca_scatter.png")
    plot_pca_scatter(df, subject_col, feature_cols, mni_idx, out_png=pca_png, standardize=standardize)

    with open(os.path.join(outdir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Outputs generated by task6_mni_comparison.py\n")
        f.write(f"- distances.csv          : per-subject distances to MNI (euclidean, cosine)\n")
        f.write(f"- distances_ranked.png   : ranked bar plot using '{metric}' distance\n")
        f.write(f"- pca_scatter.png        : 2D PCA showing MNI vs population\n")
        f.write(f"- features_used.txt      : list of numeric columns treated as features\n")

    return outdir


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task6: Compare MNI subject to population and visualize.")
    parser.add_argument("--csv", dest="csv_path", required=True, help="Path to input CSV file.")
    parser.add_argument("--subject-col", dest="subject_col", default="subject", help="Column with subject IDs (default: subject).")
    parser.add_argument("--mni-id", dest="mni_id", default=None, help="Explicit subject ID for the MNI subject (optional).")
    parser.add_argument("--outdir", dest="outdir", default="./task6_results", help="Output directory (default: ./task6_results).")
    parser.add_argument("--no-standardize", action="store_true", help="Disable feature standardization before distances/PCA.")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean", help="Metric for ranked bar plot (default: euclidean).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    out = run(
        csv_path=args.csv_path,
        subject_col=args.subject_col,
        mni_id=args.mni_id,
        outdir=args.outdir,
        standardize=not args.no_standardize,
        metric=args.metric
    )
    print(f"Done. Results saved to: {out}")


if __name__ == "__main__":
    main()
