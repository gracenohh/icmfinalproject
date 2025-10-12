import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, rankdata
import matplotlib.pyplot as plt

# --------------------------
# Utilities
# --------------------------
def strip_quotes(s):
    if isinstance(s, str):
        return s.strip().strip("'\"")
    return s

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def benjamini_hochberg(pvals: np.ndarray):
    """Return BH/FDR-adjusted q-values for a 1D array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked_p = p[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked_p[i] * n / rank
        if val > prev:
            val = prev
        prev = val
        adj[i] = val
    q = np.empty(n, dtype=float)
    q[order] = adj
    return np.clip(q, 0, 1)

def compute_rank_stats(a: np.ndarray, c: np.ndarray):
    """
    Compute rank statistics for two groups.
    Returns mean ranks and rank difference (a_mean_rank - c_mean_rank).
    """
    if len(a) == 0 or len(c) == 0:
        return np.nan, np.nan, np.nan
    
    # Combine both groups
    combined = np.concatenate([a, c])
    
    # Rank all values together (average ranking for ties)
    ranks = rankdata(combined, method='average')
    
    # Split ranks back to original groups
    n_a = len(a)
    a_ranks = ranks[:n_a]
    c_ranks = ranks[n_a:]
    
    # Calculate mean ranks
    a_mean_rank = np.mean(a_ranks)
    c_mean_rank = np.mean(c_ranks)
    rank_diff = a_mean_rank - c_mean_rank
    
    return float(a_mean_rank), float(c_mean_rank), float(rank_diff)

def cliffs_delta(a: np.ndarray, b: np.ndarray):
    """
    Cliff's delta (nonparametric effect size):
      delta = P(a > b) - P(a < b)
    Returns (delta, magnitude_label).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan, "NA"
    # Efficient computation via sorting
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    i = j = gt = lt = 0
    while i < n1 and j < n2:
        if a_sorted[i] > b_sorted[j]:
            gt += (n1 - i)  # all remaining a[i:] > b[j]
            j += 1
        elif a_sorted[i] < b_sorted[j]:
            lt += (n2 - j)  # all remaining b[j:] > a[i]
            i += 1
        else:
            # Handle ties: advance both while equal
            a_val = a_sorted[i]
            na = 0
            nb = 0
            while i < n1 and a_sorted[i] == a_val:
                i += 1; na += 1
            while j < n2 and b_sorted[j] == a_val:
                j += 1; nb += 1
            # ties contribute neither to gt nor lt
    delta = (gt - lt) / (n1 * n2)
    # Magnitude thresholds per common conventions
    ad = abs(delta)
    if ad < 0.147:
        mag = "negligible"
    elif ad < 0.33:
        mag = "small"
    elif ad < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return float(delta), mag

def find_diagnosis_column(df: pd.DataFrame) -> str:
    cand = [c for c in df.columns if "diagnosis" in c.lower()]
    if cand:
        return cand[0]
    cand = [c for c in df.columns if "group" in c.lower()]
    if cand:
        return cand[0]
    raise ValueError("Could not find a 'Diagnosis' (or 'Group') column.")

def choose_groups(df: pd.DataFrame, diagnosis_col: str):
    labels = df[diagnosis_col].astype(str)
    adhd_mask = labels.str.contains("ADHD", case=False, na=False)
    ctrl_mask = labels.str.contains("control|normal", case=False, na=False)
    return df[adhd_mask].copy(), df[ctrl_mask].copy()

def sig_stars(p_or_q: float):
    if pd.isna(p_or_q):
        return ""
    if p_or_q < 0.001:
        return "***"
    if p_or_q < 0.01:
        return "**"
    if p_or_q < 0.05:
        return "*"
    return "ns"

def plot_box_for_roi(df: pd.DataFrame, diagnosis_col: str, roi: str, outdir: Path,
                     mw_p: float = np.nan, mw_q: float = np.nan, 
                     rank_diff: float = np.nan):
    labels = df[diagnosis_col].astype(str)
    adhd = df[labels.str.contains("ADHD", case=False, na=False)][roi].dropna().values
    ctrl  = df[labels.str.contains("control|normal", case=False, na=False)][roi].dropna().values

    fig = plt.figure(figsize=(6, 5))
    # Matplotlib 3.9+: labels -> tick_labels
    plt.boxplot([ctrl, adhd], tick_labels=["Control/Normal", "ADHD"])
    star = sig_stars(mw_q if not pd.isna(mw_q) else mw_p)
    
    # Enhanced title with rank information
    title_line1 = f"{roi} volume: ADHD vs Control"
    title_line2 = f"MW p={mw_p:.3g}, q={mw_q:.3g} {star}" if not pd.isna(mw_p) else ""
    title_line3 = f"Rank diff (ADHD-Control): {rank_diff:+.1f}" if not pd.isna(rank_diff) else ""
    
    full_title = title_line1
    if title_line2:
        full_title += "\n" + title_line2
    if title_line3:
        full_title += "\n" + title_line3
    
    plt.title(full_title, fontsize=10)
    plt.xlabel("Group")
    plt.ylabel(roi)
    plt.tight_layout()
    outpath = outdir / f"{sanitize_filename(roi)}_boxplot_MW.png"
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    return outpath

# --------------------------
# Main
# --------------------------
def main(roi_csv: str, excel_path: str, outdir: str, alpha: float = 0.05):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ROIs
    roi_df = pd.read_csv(roi_csv)
    roi_colname = roi_df.columns[0]
    rois = [strip_quotes(str(x)) for x in roi_df[roi_colname].dropna().tolist()]

    # Data
    df = pd.read_excel(excel_path)
    df.columns = [strip_quotes(c) for c in df.columns]
    # Pandas 2.1+: DataFrame.map applies elementwise
    df = df.map(strip_quotes)

    diagnosis_col = find_diagnosis_column(df)
    adhd_df, ctrl_df = choose_groups(df, diagnosis_col)

    # First pass: compute MW p-values, effect sizes, and rank statistics
    rows = []
    skipped = []
    for roi in rois:
        if roi not in df.columns:
            skipped.append(roi)
            continue

        a = adhd_df[roi].dropna().astype(float).values
        c = ctrl_df[roi].dropna().astype(float).values
        if len(a) < 2 or len(c) < 2:
            rows.append({
                "ROI": roi,
                "ADHD_n": len(a), "Control_n": len(c),
                "ADHD_mean": float(np.mean(a)) if len(a) else np.nan,
                "Control_mean": float(np.mean(c)) if len(c) else np.nan,
                "ADHD_mean_rank": np.nan,
                "Control_mean_rank": np.nan,
                "Rank_diff": np.nan,
                "MW_U": np.nan, "MW_pvalue": np.nan,
                "Cliffs_delta": np.nan, "Cliffs_magnitude": "NA",
                "Note": "Insufficient samples",
                "Plot": ""
            })
            continue

        # MW test (SciPy chooses exact/asymptotic automatically)
        U, p_u = mannwhitneyu(a, c, alternative="two-sided", method="auto")
        delta, delta_mag = cliffs_delta(a, c)
        
        # Compute rank statistics
        a_mean_rank, c_mean_rank, rank_diff = compute_rank_stats(a, c)

        rows.append({
            "ROI": roi,
            "ADHD_n": int(len(a)), "Control_n": int(len(c)),
            "ADHD_mean": float(np.mean(a)), "Control_mean": float(np.mean(c)),
            "ADHD_mean_rank": a_mean_rank,
            "Control_mean_rank": c_mean_rank,
            "Rank_diff": rank_diff,
            "MW_U": float(U), "MW_pvalue": float(p_u),
            "Cliffs_delta": float(delta), "Cliffs_magnitude": delta_mag
        })

    res_df = pd.DataFrame(rows)

    # Add BH/FDR q-values and significance flag (kept simple & useful)
    if "MW_pvalue" in res_df.columns:
        res_df["MW_qvalue"] = benjamini_hochberg(res_df["MW_pvalue"].values)
        res_df["sig_MW_FDR_0.05"] = res_df["MW_qvalue"] < alpha
    else:
        res_df["MW_qvalue"] = np.nan
        res_df["sig_MW_FDR_0.05"] = False

    # Second pass: generate plots with p/q and rank difference annotations
    plot_paths = []
    for i, row in res_df.iterrows():
        roi = row["ROI"]
        if roi in skipped:
            plot_paths.append("")
            continue
        p = row["MW_pvalue"]
        q = row["MW_qvalue"]
        rank_diff = row["Rank_diff"]
        plot_path = plot_box_for_roi(df, diagnosis_col, roi, outdir, 
                                     mw_p=p, mw_q=q, rank_diff=rank_diff)
        plot_paths.append(str(plot_path))
    res_df["Plot"] = plot_paths

    # Save results (MW-only)
    res_csv = outdir / "basal_ganglia_ADHD_vs_Control_MW_results.csv"
    res_df.to_csv(res_csv, index=False)

    # Report
    report_txt = outdir / "report.txt"
    with report_txt.open("w", encoding="utf-8") as f:
        f.write("Basal Ganglia ROI analysis (ADHD vs Control/Normal) — MW with Rank Stats\n")
        f.write(f"Diagnosis column: {diagnosis_col}\n")
        f.write(f"Results CSV: {res_csv}\n")
        f.write(f"FDR alpha: {alpha}\n\n")
        
        f.write("Rank Difference Interpretation:\n")
        f.write("  - Positive rank diff: ADHD group has higher values on average\n")
        f.write("  - Negative rank diff: Control group has higher values on average\n")
        f.write("  - Larger absolute value: Stronger separation between groups\n\n")
        
        if skipped:
            f.write("ROIs not found in Excel columns:\n")
            for r in skipped:
                f.write(f"  - {r}\n")

    print(f"[OK] Saved MW results with rank statistics to: {res_csv}")
    if skipped:
        print(f"[WARN] Missing ROIs (not in Excel): {', '.join(skipped)}")
    print(f"[OK] Per-ROI boxplots (with MW p/q, stars & rank diff) saved under: {outdir}")

if __name__ == "__main__":
    # Update these paths or add argparse if you prefer CLI flags
    main('/Users/harry/Desktop/ICM/Final Proect/basal_ganglia_rois.csv',
         '/Users/harry/Desktop/ICM/Final Proect/ADHD_ICM_random200.xlsx',
         '/Users/harry/Desktop/ICM/Final Proect/')