"""
Dataset loading, cleaning, and subset extraction for the LOPAC1280 drug screen.

The master CSV contains 3,072 rows (1,280 unique drugs + 512 controls,
across 4 plates x 2 replicates). This module provides:
  - Raw loading and text cleaning
  - Unique drug extraction (1,280 compounds)
  - Biologically meaningful subset extraction (PC4, PC7, PC2, pilot30)
"""

import pandas as pd
import config


def load_raw_dataset() -> pd.DataFrame:
    """Load the master CSV dataset."""
    df = pd.read_csv(config.DATASET_CSV)
    print(f"Loaded {len(df)} rows from dataset")
    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text columns: strip whitespace, embedded newlines, and normalize
    Action values to title case.
    """
    df = df.copy()
    text_cols = ["compound_clean", "compound_symbol", "Class", "Action",
                 "Selectivity", "Description"]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r"[\n\r\t]+", " ", regex=True)
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)
            # Replace string 'nan' back to actual NaN
            df[col] = df[col].replace("nan", pd.NA)
            df[col] = df[col].replace("", pd.NA)

    # Normalize Action: title case + merge known variants
    if "Action" in df.columns:
        df["Action"] = df["Action"].str.title()
        action_merge = {
            "Inhibition": "Inhibitor",
            "Antagonists": "Antagonist",
            "Channel Blocker": "Blocker",
            "Reuptake": "Inhibitor",
            "Transport Inhibitor": "Inhibitor",
            "Agonist (Inverse)": "Inverse Agonist",
            "Agonist - Antagonist": "Modulator",
            "Modulator/Inhibitor": "Modulator",
        }
        df["Action"] = df["Action"].replace(action_merge)

    return df


def extract_unique_drugs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out control wells and return deduplicated drug records.

    Returns a DataFrame with 1,280 unique drugs and their metadata.
    """
    # Filter controls
    drugs_df = df[~df["is_control"]].copy()
    print(f"Non-control rows: {len(drugs_df)}")

    # Deduplicate on compound_clean, keeping first occurrence
    unique_drugs = drugs_df.drop_duplicates(subset="compound_clean")[
        ["compound_clean", "compound_symbol", "CATNUM",
         "Class", "Action", "Selectivity", "Description",
         "mol weight Structure"]
    ].reset_index(drop=True)

    # Fill missing Action/Selectivity with "Unknown" for downstream use
    unique_drugs["Action"] = unique_drugs["Action"].fillna("Unknown")
    unique_drugs["Selectivity"] = unique_drugs["Selectivity"].fillna("Unknown")

    print(f"Unique drugs extracted: {len(unique_drugs)}")
    return unique_drugs


def get_subset(unique_drugs: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Return a named subset of the drug list.

    Subsets:
      - "pc4": Glutamate antagonists (PhenoCluster 4 - key discovery)
      - "pc7": VEGF inhibitors (PhenoCluster 7 - gold standard anti-angiogenic)
      - "pc2": Random sample of 20 mixed drugs (control-like baseline)
      - "pilot30": Union of 10 from each of PC4, PC7, PC2
      - "all": Full 1,280 drugs
    """
    name = name.lower()

    if name == "all":
        return unique_drugs

    if name == "pc4":
        # Glutamate class + Antagonist action
        mask = (
            (unique_drugs["Class"].str.lower().str.contains("glutamate", na=False))
            & (unique_drugs["Action"].str.lower().str.contains("antagonist", na=False))
        )
        subset = unique_drugs[mask].reset_index(drop=True)
        print(f"PC4 (Glutamate antagonists): {len(subset)} drugs")
        return subset

    if name == "pc7":
        # VEGF inhibitors - match by name, description, or selectivity
        vegf_names_lower = [v.lower() for v in config.PC7_VEGF_INHIBITORS]
        name_mask = unique_drugs["compound_clean"].str.lower().isin(vegf_names_lower)
        desc_mask = unique_drugs["Description"].str.lower().str.contains(
            "vegf|vascular endothelial growth factor|vegfr|kdr|angiogenesis",
            na=False, regex=True,
        )
        sel_mask = unique_drugs["Selectivity"].str.lower().str.contains(
            "vegf|vegfr|kdr", na=False, regex=True
        )
        mask = name_mask | desc_mask | sel_mask
        subset = unique_drugs[mask].reset_index(drop=True)
        print(f"PC7 (VEGF inhibitors): {len(subset)} drugs")
        return subset

    if name == "pc2":
        # Random sample of 20 mixed-class drugs (control-like)
        # Exclude known PC4/PC7 drugs
        pc4 = get_subset(unique_drugs, "pc4")
        pc7 = get_subset(unique_drugs, "pc7")
        exclude = set(pc4["compound_clean"].tolist() + pc7["compound_clean"].tolist())
        candidates = unique_drugs[~unique_drugs["compound_clean"].isin(exclude)]
        subset = candidates.sample(n=min(20, len(candidates)),
                                   random_state=config.EVAL_SEED)
        subset = subset.reset_index(drop=True)
        print(f"PC2 (control-like sample): {len(subset)} drugs")
        return subset

    if name == "pilot30":
        pc4 = get_subset(unique_drugs, "pc4").head(10)
        pc7 = get_subset(unique_drugs, "pc7").head(10)
        pc2 = get_subset(unique_drugs, "pc2").head(10)
        subset = pd.concat([pc4, pc7, pc2], ignore_index=True)
        # Remove any duplicates
        subset = subset.drop_duplicates(subset="compound_clean").reset_index(drop=True)
        print(f"Pilot30: {len(subset)} drugs")
        return subset

    raise ValueError(
        f"Unknown subset '{name}'. "
        f"Available: all, pc4, pc7, pc2, pilot30"
    )


def save_drug_list(df: pd.DataFrame, path=None):
    """Save drug list to CSV in the phase1 output directory."""
    if path is None:
        path = config.PHASE1_DIR / "drug_master_list.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} drugs to {path}")


def get_drug_metadata_row(drug_row: pd.Series) -> dict:
    """Convert a drug DataFrame row to a metadata dict for LLM prompting."""
    return {
        "name": str(drug_row["compound_clean"]),
        "class": str(drug_row.get("Class", "Unknown")),
        "action": str(drug_row.get("Action", "Unknown")),
        "selectivity": str(drug_row.get("Selectivity", "Unknown")),
        "description": str(drug_row.get("Description", "")),
    }


def print_dataset_stats(unique_drugs: pd.DataFrame):
    """Print summary statistics of the drug dataset."""
    print("\n" + "=" * 50)
    print("  LOPAC1280 Drug Dataset Summary")
    print("=" * 50)
    print(f"  Total unique drugs: {len(unique_drugs)}")

    print("\n  Top 10 Drug Classes:")
    class_counts = unique_drugs["Class"].value_counts().head(10)
    for cls, cnt in class_counts.items():
        print(f"    {cls:30s} {cnt:4d}")

    print("\n  Action Distribution:")
    action_counts = unique_drugs["Action"].value_counts()
    for act, cnt in action_counts.items():
        print(f"    {act:30s} {cnt:4d}")

    print("\n  Subset Sizes:")
    for name in ["pc4", "pc7", "pc2", "pilot30"]:
        try:
            subset = get_subset(unique_drugs, name)
            print(f"    {name:10s}: {len(subset)} drugs")
        except Exception:
            pass

    print("=" * 50)


if __name__ == "__main__":
    df = load_raw_dataset()
    df = clean_text_columns(df)
    unique_drugs = extract_unique_drugs(df)
    print_dataset_stats(unique_drugs)
    save_drug_list(unique_drugs)
