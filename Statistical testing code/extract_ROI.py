import pandas as pd
from pathlib import Path

def extract_basal_ganglia_rois(path: str):
    fp = Path(path)

    # Load (handles either tab or multi-space separators)
    try:
        df = pd.read_csv(fp, sep="\t", engine="python")
        if df.columns[0] != "Type1-L5":
            df.columns = ["ID","Type1-L5","Type1-L4","Type1-L3","Type1-L2","Type1-L1",
                          "Type2-L5","Type2-L4","Type2-L3","Type2-L2","Type2-L1"]
        else:
            df.insert(0, "ID", range(1, len(df) + 1))
    except Exception:
        df = pd.read_csv(fp, sep=r"\s{2,}|\t", engine="python", header=0)
        if df.shape[1] == 10:
            df.insert(0, "ID", range(1, len(df) + 1))
        if df.shape[1] == 11:
            df.columns = ["ID","Type1-L5","Type1-L4","Type1-L3","Type1-L2","Type1-L1",
                          "Type2-L5","Type2-L4","Type2-L3","Type2-L2","Type2-L1"]

    # Match any row where any column includes "BasalGang"
    mask = df.apply(lambda s: s.astype(str).str.contains("BasalGang", case=False, na=False))
    basal_rows = df[mask.any(axis=1)].copy()

    # ROI short names live in "Type1-L5"
    rois = sorted(basal_rows["Type1-L5"].astype(str).unique())

    # Save outputs
    txt_path = fp.with_name("basal_ganglia_rois.txt")
    csv_path = fp.with_name("basal_ganglia_rois.csv")
    txt_path.write_text("\n".join(rois), encoding="utf-8")
    pd.Series(rois, name="BasalGangliaROIs").to_csv(csv_path, index=False)

    return rois, basal_rows

if __name__ == "__main__":
    rois, _ = extract_basal_ganglia_rois("/Users/harry/Desktop/ICM/Final Proect/multilevel_lookup_table.txt")
    print("Basal ganglia ROIs ({}):".format(len(rois)))
    for r in rois:
        print(r)
