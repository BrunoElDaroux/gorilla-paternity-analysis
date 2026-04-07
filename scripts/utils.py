"""
utils.py
========
Shared utilities for the gorilla paternity analysis pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Project path resolution ───────────────────────────────────────────────────


def get_project_root() -> Path:
    """
    Resolve project root robustly whether called from:
      - scripts/ directory
      - notebooks/ directory
      - project root
    """
    cwd = Path(os.getcwd())
    if cwd.name in ("scripts", "notebooks"):
        return cwd.parent
    return cwd


def add_scripts_to_path() -> None:
    """Add scripts/ directory to sys.path for importing custom modules."""
    root = get_project_root()
    scripts = str(root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def get_data_dir(subdir: str = "raw") -> Path:
    root = get_project_root()
    d = root / "data" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_output_dir(subdir: str = "figures") -> Path:
    root = get_project_root()
    d = root / "outputs" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Data loaders ──────────────────────────────────────────────────────────────


def load_all_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Load all raw CSVs into a dict of DataFrames."""
    if data_dir is None:
        data_dir = get_data_dir("raw")

    files = {
        "individuals": "individuals.csv",
        "genotypes": "genotypes.csv",
        "genotypes_true": "genotypes_true.csv",
        "allele_frequencies": "allele_frequencies.csv",
        "groups": "groups.csv",
    }
    out = {}
    for key, fname in files.items():
        fpath = data_dir / fname
        if fpath.exists():
            out[key] = pd.read_csv(fpath)
        else:
            print(f"  [WARNING] {fname} not found — run data_generator.py first")
    return out


LOCUS_NAMES = [
    "D2S1338",
    "D3S1358",
    "D5S818",
    "D7S820",
    "D8S1179",
    "D13S317",
    "D16S539",
    "D18S51",
    "D19S433",
    "D21S11",
    "vWA",
    "FGA",
    "CSF1PO",
    "TPOX",
    "TH01",
]

AGE_CLASS_ORDER = ["infant", "juvenile", "adult", "silverback"]
COLOUR_MAP = {
    "silverback": "#2C3E50",
    "adult": "#E67E22",
    "juvenile": "#27AE60",
    "infant": "#3498DB",
}
GROUP_PALETTE = sns.color_palette("Set2", 6)


# ── Genotype helpers ──────────────────────────────────────────────────────────


def missing_data_summary(geno_df: pd.DataFrame, locus_names: List[str]) -> pd.DataFrame:
    """
    Compute per-locus and per-individual missing data rates.

    Returns DataFrame: locus | n_missing | pct_missing | n_total
    """
    rows = []
    n_total = len(geno_df)

    for locus in locus_names:
        a1_miss = geno_df[f"{locus}_a1"].isna()
        a2_miss = geno_df[f"{locus}_a2"].isna()
        n_miss = int((a1_miss | a2_miss).sum())
        rows.append(
            {
                "locus": locus,
                "n_missing": n_miss,
                "pct_missing": round(100 * n_miss / n_total, 2),
                "n_total": n_total,
            }
        )

    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False)


def individual_missing_data(
    geno_df: pd.DataFrame, locus_names: List[str]
) -> pd.DataFrame:
    """Compute per-individual missing locus proportion."""
    rows = []
    for _, row in geno_df.iterrows():
        n_miss = sum(
            1
            for l in locus_names
            if pd.isna(row.get(f"{l}_a1")) or pd.isna(row.get(f"{l}_a2"))
        )
        rows.append(
            {
                "individual_id": row["individual_id"],
                "n_loci_missing": n_miss,
                "pct_missing": round(100 * n_miss / len(locus_names), 2),
            }
        )
    return pd.DataFrame(rows)


def missing_data_matrix(geno_df: pd.DataFrame, locus_names: List[str]) -> pd.DataFrame:
    """
    Build binary matrix: individuals × loci, 1 = missing, 0 = present.
    """
    mat_rows = []
    for _, row in geno_df.iterrows():
        mat_row = {"individual_id": row["individual_id"]}
        for locus in locus_names:
            missing = pd.isna(row.get(f"{locus}_a1")) or pd.isna(row.get(f"{locus}_a2"))
            mat_row[locus] = int(missing)
        mat_rows.append(mat_row)
    return pd.DataFrame(mat_rows).set_index("individual_id")


# ── Plotting helpers ──────────────────────────────────────────────────────────


def set_plot_style():
    """Apply consistent, publication-ready matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
        }
    )


def save_figure(
    fig: plt.Figure,
    name: str,
    subdir: str = "figures",
    dpi: int = 200,
    tight: bool = True,
) -> Path:
    """Save figure to outputs directory."""
    out_dir = get_output_dir(subdir)
    fpath = out_dir / f"{name}.png"
    if tight:
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(fpath, dpi=dpi, facecolor="white")
    print(f"  Saved → {fpath}")
    return fpath


# ── Statistical helpers ───────────────────────────────────────────────────────


def mannwhitney_dominance_paternity(
    paternity_df: pd.DataFrame, individuals_df: pd.DataFrame
) -> Dict:
    """
    Test whether assigned fathers' dominance rank differs from what's expected
    under random mating.

    Compares dominance rank of actual sires vs all candidate males.
    Lower rank = higher dominance (rank 1 = dominant).
    """
    from scipy.stats import mannwhitneyu

    male_df = individuals_df[individuals_df["sex"] == "M"][
        ["individual_id", "dominance_rank", "group_id"]
    ].dropna()

    assigned = paternity_df[paternity_df["assigned_father"].notna()]
    sire_ranks = []
    for _, row in assigned.iterrows():
        rank_row = male_df[male_df["individual_id"] == row["assigned_father"]]
        if not rank_row.empty:
            sire_ranks.append(int(rank_row["dominance_rank"].values[0]))

    all_ranks = male_df["dominance_rank"].dropna().astype(int).tolist()

    if not sire_ranks:
        return {"error": "No assigned fathers found in individual metadata"}

    stat, p = mannwhitneyu(sire_ranks, all_ranks, alternative="less")
    return {
        "n_assigned": len(sire_ranks),
        "mean_sire_rank": round(np.mean(sire_ranks), 2),
        "mean_all_rank": round(np.mean(all_ranks), 2),
        "U_statistic": stat,
        "p_value": round(float(p), 4),
        "significant": float(p) < 0.05,
        "interpretation": (
            "Sires have significantly lower (higher) rank "
            "than random males — dominance predicts paternity."
            if float(p) < 0.05
            else "No significant dominance-paternity association."
        ),
    }


def accuracy_vs_truth(paternity_df: pd.DataFrame, ind_df: pd.DataFrame) -> Dict:
    """
    Compare assigned fathers to ground-truth fathers.
    Only meaningful for simulated data.
    """
    truth = ind_df[["individual_id", "true_father_id"]].set_index("individual_id")
    merged = paternity_df.merge(
        truth, left_on="offspring_id", right_index=True, how="left"
    )
    has_truth = merged[
        merged["true_father_id"].notna() & (merged["true_father_id"] != "UNSAMPLED")
    ]
    if has_truth.empty:
        return {"error": "No truth labels available"}

    correct = (has_truth["assigned_father"] == has_truth["true_father_id"]).sum()
    return {
        "n_offspring": len(has_truth),
        "n_correct": int(correct),
        "accuracy": round(correct / len(has_truth), 4),
        "n_assigned": int(has_truth["assigned_father"].notna().sum()),
        "n_unresolved": int(has_truth["assigned_father"].isna().sum()),
    }
