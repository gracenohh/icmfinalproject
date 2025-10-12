import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def extract_basal_ganglia_indices(lookup_table_path: str):
    """Extract basal ganglia ROI indices from multilevel lookup table."""
    fp = Path(lookup_table_path)
    
    # Load the lookup table
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
    
    # Find rows with "BasalGang" in any column
    mask = df.apply(lambda s: s.astype(str).str.contains("BasalGang", case=False, na=False))
    basal_rows = df[mask.any(axis=1)].copy()
    
    # Extract ROI names and their corresponding indices (ID - 1 since volumes are 0-indexed)
    basal_ganglia_info = []
    for _, row in basal_rows.iterrows():
        roi_name = row["Type1-L5"]
        roi_index = row["ID"] - 1  # Convert to 0-based indexing for volume data
        basal_ganglia_info.append({
            'name': roi_name,
            'index': roi_index,
            'id': row["ID"]
        })
    
    return basal_ganglia_info, basal_rows

def load_mni_volume_data(volume_data_path: str):
    """Load MNI volume data."""
    df = pd.read_csv(volume_data_path, sep="\t")
    return df

def create_mni_barcode(lookup_table_path: str, volume_data_path: str, output_dir: str = "."):
    """Create barcode visualization for MNI subject based on basal ganglia ROIs."""
    
    # Step 1: Extract basal ganglia ROI information
    print("Extracting basal ganglia ROI indices...")
    basal_ganglia_info, basal_rows = extract_basal_ganglia_indices(lookup_table_path)
    
    print(f"Found {len(basal_ganglia_info)} basal ganglia ROIs:")
    for info in basal_ganglia_info:
        print(f"  - {info['name']} (ID: {info['id']}, Index: {info['index']})")
    
    # Step 2: Load MNI volume data
    print("\nLoading MNI volume data...")
    volume_df = load_mni_volume_data(volume_data_path)
    
    # Step 3: Extract basal ganglia volumes
    basal_ganglia_volumes = []
    basal_ganglia_names = []
    
    for info in basal_ganglia_info:
        # Find the volume data row that matches our ROI index
        roi_row = volume_df[volume_df['Label Id'] == info['id']]
        if not roi_row.empty:
            volume = roi_row['Volume (mm^3)'].iloc[0]
            basal_ganglia_volumes.append(volume)
            basal_ganglia_names.append(info['name'])
            print(f"  - {info['name']}: {volume} mm³")
    
    return basal_ganglia_df, fig

if __name__ == "__main__":
    # Define file paths
    lookup_table_path = "multilevel_lookup_table.txt"
    volume_data_path = "Volume_MNI_ICM_Final_Project.txt"
    
    # Create the barcode
    basal_ganglia_df, fig = create_mni_barcode(lookup_table_path, volume_data_path)
