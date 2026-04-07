"""
kinship_functions.py
====================
Pairwise genetic relatedness estimators for microsatellite data.

Implements three estimators commonly used in wild primate studies:

1. Proportion of Alleles Shared (PAS / IBS)
   - Simple, non-parametric; ranges 0–1
   - Full sibs/parent-offspring ≈ 0.5; unrelated ≈ 0.0 (after correction)

2. Queller & Goodnight (1989) rxy
   - rxy = (Σ_l (p_ia - p_a)(p_ja - p_a)) / (Σ_l p_ia(1-p_a))
   - Asymmetric; must be symmetrised as (rxy + ryx)/2
   - Expected: parent-offspring = 0.5, full sibs = 0.5, half sibs = 0.25, unrelated = 0

3. Lynch & Ritland (1999) r̂
   - Method-of-moments using multi-allelic data
   - Often lower variance than QG for small sample sizes

References
----------
Queller DC, Goodnight KF (1989) Evolution 43:258-275.
Lynch M, Ritland K (1999) Genetics 152:1753-1766.
Wang J (2002) Genetics 160:1203-1215.
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Genotype dictionary builder ───────────────────────────────────────────────


def build_geno_dict(
    geno_df: pd.DataFrame, locus_names: List[str]
) -> Dict[str, Dict[str, list]]:
    """
    Convert wide-format genotype DataFrame to nested dict:
    {individual_id: {locus: [a1, a2]}}
    Missing values stored as None.
    """
    geno_dict = {}
    for _, row in geno_df.iterrows():
        iid = row["individual_id"]
        geno = {}
        for locus in locus_names:
            a1 = row.get(f"{locus}_a1")
            a2 = row.get(f"{locus}_a2")
            a1 = (
                None if (pd.isna(a1) if not isinstance(a1, str) else False) else int(a1)
            )
            a2 = (
                None if (pd.isna(a2) if not isinstance(a2, str) else False) else int(a2)
            )
            geno[locus] = [a1, a2]
        geno_dict[iid] = geno
    return geno_dict


# ── Proportion of alleles shared (IBS) ───────────────────────────────────────


def pas_single_locus(
    g1: List[Optional[int]], g2: List[Optional[int]]
) -> Optional[float]:
    """
    Proportion of alleles shared by identity of state at one locus.
    Returns None if either genotype is missing.

    PAS = (I(a==c) + I(a==d) + I(b==c) + I(b==d)) / 4
    where {a,b} = g1, {c,d} = g2.
    """
    a1, a2 = g1
    b1, b2 = g2
    if any(x is None for x in [a1, a2, b1, b2]):
        return None
    shared = int(a1 == b1) + int(a1 == b2) + int(a2 == b1) + int(a2 == b2)
    return shared / 4.0


def pas_multilocus(id1: str, id2: str, geno_dict: Dict, locus_names: List[str]) -> Dict:
    """Multi-locus PAS (mean over informative loci)."""
    g1 = geno_dict.get(id1, {})
    g2 = geno_dict.get(id2, {})

    values = []
    n_missing = 0

    for locus in locus_names:
        v = pas_single_locus(g1.get(locus, [None, None]), g2.get(locus, [None, None]))
        if v is None:
            n_missing += 1
        else:
            values.append(v)

    if not values:
        return {"pas": np.nan, "n_loci": 0, "n_missing": n_missing}

    return {
        "pas": round(float(np.mean(values)), 6),
        "n_loci": len(values),
        "n_missing": n_missing,
    }


# ── Queller & Goodnight (1989) rxy ───────────────────────────────────────────


def qg_r_single_locus(
    g_focal: List[Optional[int]],
    g_other: List[Optional[int]],
    locus_freqs: Dict[int, float],
) -> Optional[Tuple[float, float]]:
    """
    Compute numerator and denominator of QG rxy at a single locus.
    Returns (numerator, denominator) or None if data are missing.

    rxy = Σ_a (p_ia - p_a)(p_ja - p_a) / Σ_a p_ia(1-p_a)
    where p_ia = 1 if focal individual carries allele a, 0.5 if heterozygous,
          p_a  = population frequency of allele a.

    Per-locus contribution is accumulated across loci then divided.
    """
    a1, a2 = g_focal
    b1, b2 = g_other

    if any(x is None for x in [a1, a2, b1, b2]):
        return None

    # All alleles at this locus (union)
    all_alleles = set([a1, a2, b1, b2]) | set(locus_freqs.keys())

    num = 0.0
    den = 0.0

    for allele in all_alleles:
        p_a = locus_freqs.get(allele, 0.005)  # small pseudocount

        # p_ia: allele sharing indicator for focal individual
        if a1 == a2 == allele:  # homozygous
            p_ia = 1.0
        elif allele in (a1, a2):  # heterozygous carrier
            p_ia = 0.5
        else:
            p_ia = 0.0

        # p_ja: for other individual
        if b1 == b2 == allele:
            p_ja = 1.0
        elif allele in (b1, b2):
            p_ja = 0.5
        else:
            p_ja = 0.0

        num += (p_ia - p_a) * (p_ja - p_a)
        den += p_ia * (1.0 - p_a)

    return (num, den)


def qg_relatedness(
    id1: str,
    id2: str,
    geno_dict: Dict,
    freq_table: Dict[str, Dict[int, float]],
    locus_names: List[str],
) -> Dict:
    """
    Multi-locus Queller-Goodnight rxy, symmetrised as (rxy + ryx) / 2.
    Accumulates numerators and denominators before dividing (recommended).
    """
    g1 = geno_dict.get(id1, {})
    g2 = geno_dict.get(id2, {})

    num_xy, den_xy = 0.0, 0.0
    num_yx, den_yx = 0.0, 0.0
    n_loci = 0

    for locus in locus_names:
        lf = freq_table.get(locus, {})
        res_xy = qg_r_single_locus(
            g1.get(locus, [None, None]), g2.get(locus, [None, None]), lf
        )
        res_yx = qg_r_single_locus(
            g2.get(locus, [None, None]), g1.get(locus, [None, None]), lf
        )

        if res_xy and res_yx:
            num_xy += res_xy[0]
            den_xy += res_xy[1]
            num_yx += res_yx[0]
            den_yx += res_yx[1]
            n_loci += 1

    rxy = (num_xy / den_xy) if den_xy != 0 else np.nan
    ryx = (num_yx / den_yx) if den_yx != 0 else np.nan

    if np.isnan(rxy) or np.isnan(ryx):
        r_sym = np.nan
    else:
        r_sym = (rxy + ryx) / 2.0

    return {
        "r_qg": round(r_sym, 6),
        "n_loci": n_loci,
    }


# ── Lynch & Ritland (1999) r̂ ─────────────────────────────────────────────────


def lr_relatedness(
    id1: str,
    id2: str,
    geno_dict: Dict,
    freq_table: Dict[str, Dict[int, float]],
    locus_names: List[str],
) -> Dict:
    """
    Lynch & Ritland (1999) pairwise relatedness estimator r̂.

    For a pair with genotypes (a,b) and (c,d):
      r̂_locus = [p_a(S_bc + S_bd) + p_b(S_ac + S_ad) - 4p_a*p_b]
               / [(1 + S_ab)(p_a + p_b) - 4p_a*p_b]

    where S_ij = 1 if i==j else 0, and alleles a,b are those of individual X.

    Averaged over loci (weighted by denominator).
    """
    g1 = geno_dict.get(id1, {})
    g2 = geno_dict.get(id2, {})

    weighted_num = 0.0
    weighted_den = 0.0
    n_loci = 0

    for locus in locus_names:
        lf = freq_table.get(locus, {})
        a, b = g1.get(locus, [None, None])
        c, d = g2.get(locus, [None, None])

        if any(x is None for x in [a, b, c, d]):
            continue

        # LR estimator uses alleles of individual X (a,b) as reference
        # Symmetrise by averaging over both orientations
        def _lr_directed(a, b, c, d, lf):
            p_a = lf.get(a, 0.005)
            p_b = lf.get(b, 0.005)
            S_ab = int(a == b)
            S_ac = int(a == c)
            S_ad = int(a == d)
            S_bc = int(b == c)
            S_bd = int(b == d)
            num = p_a * (S_bc + S_bd) + p_b * (S_ac + S_ad) - 4 * p_a * p_b
            den = (1 + S_ab) * (p_a + p_b) - 4 * p_a * p_b
            return num, den

        n1, d1 = _lr_directed(a, b, c, d, lf)
        n2, d2 = _lr_directed(c, d, a, b, lf)

        if d1 != 0 and d2 != 0:
            weighted_num += (n1 / d1 + n2 / d2) / 2 * ((abs(d1) + abs(d2)) / 2)
            weighted_den += (abs(d1) + abs(d2)) / 2
            n_loci += 1

    r_lr = (weighted_num / weighted_den) if weighted_den != 0 else np.nan

    return {
        "r_lr": round(float(r_lr), 6),
        "n_loci": n_loci,
    }


# ── Full pairwise relatedness matrix ─────────────────────────────────────────


def compute_pairwise_relatedness(
    ind_ids: List[str],
    geno_dict: Dict,
    freq_table: Dict,
    locus_names: List[str],
    method: str = "both",
) -> pd.DataFrame:
    """
    Compute all pairwise relatedness values for a list of individuals.

    Parameters
    ----------
    ind_ids     : list of individual IDs to include
    method      : 'qg' | 'lr' | 'pas' | 'both' (default: both QG and LR)

    Returns
    -------
    DataFrame with columns: id1, id2, r_qg, r_lr, pas, n_loci
    """
    rows = []
    pairs = list(combinations(ind_ids, 2))

    for id1, id2 in pairs:
        row = {"id1": id1, "id2": id2}

        if method in ("qg", "both"):
            qg = qg_relatedness(id1, id2, geno_dict, freq_table, locus_names)
            row.update({"r_qg": qg["r_qg"], "n_loci_qg": qg["n_loci"]})

        if method in ("lr", "both"):
            lr = lr_relatedness(id1, id2, geno_dict, freq_table, locus_names)
            row.update({"r_lr": lr["r_lr"]})

        pas = pas_multilocus(id1, id2, geno_dict, locus_names)
        row.update({"pas": pas["pas"], "n_loci_pas": pas["n_loci"]})

        rows.append(row)

    return pd.DataFrame(rows)


def relatedness_to_matrix(
    pairwise_df: pd.DataFrame, ind_ids: List[str], r_col: str = "r_qg"
) -> pd.DataFrame:
    """
    Convert pairwise long-format relatedness DataFrame to a symmetric matrix.
    Diagonal = 1.0 (self-relatedness).
    """
    arr = np.full((len(ind_ids), len(ind_ids)), np.nan)
    np.fill_diagonal(arr, 1.0)
    mat = pd.DataFrame(arr, index=ind_ids, columns=ind_ids)

    for _, row in pairwise_df.iterrows():
        i1, i2 = row["id1"], row["id2"]
        val = row[r_col]
        mat.loc[i1, i2] = val
        mat.loc[i2, i1] = val

    return mat


# ── Within- vs between-group relatedness ─────────────────────────────────────


def group_relatedness_summary(
    pairwise_df: pd.DataFrame, ind_df: pd.DataFrame, r_col: str = "r_qg"
) -> pd.DataFrame:
    """
    Compute mean pairwise relatedness within and between social groups.
    Expects ind_df to have columns: individual_id, group_id.
    """
    id_to_group = ind_df.set_index("individual_id")["group_id"].to_dict()

    records = []
    for _, row in pairwise_df.iterrows():
        g1 = id_to_group.get(row["id1"])
        g2 = id_to_group.get(row["id2"])
        if g1 is None or g2 is None:
            continue
        context = "within" if g1 == g2 else "between"
        records.append(
            {
                "id1": row["id1"],
                "id2": row["id2"],
                "group1": g1,
                "group2": g2,
                "context": context,
                "relatedness": row[r_col],
            }
        )

    rec_df = pd.DataFrame(records)
    summary = (
        rec_df.groupby("context")["relatedness"]
        .agg(["mean", "median", "std", "count"])
        .rename(
            columns={
                "mean": "mean_r",
                "median": "median_r",
                "std": "sd_r",
                "count": "n_pairs",
            }
        )
    )
    return summary
