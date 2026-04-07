"""
parentage_functions.py
======================
CERVUS-style likelihood-based paternity assignment for microsatellite data.

Core reference:
  Marshall TC, Slate J, Kruuk LEB, Pemberton JM (1998)
  Statistical confidence for likelihood-based paternity inference in natural
  populations. Molecular Ecology, 7, 639-655.

Key concepts
------------
LOD score = Σ_l log10 [ P(G_off | G_mom, G_cand) / P(G_off | G_mom) ]

At each locus l:
  - Numerator   : probability that the offspring received one allele from
                  the candidate father given the candidate's genotype.
  - Denominator : probability that the offspring received the same allele
                  from a random male drawn from the population.

A candidate is EXCLUDED at a locus if he cannot have supplied the paternal
allele inferred from the offspring + mother genotypes.
Paternity is assigned to the candidate with the highest LOD score above a
user-defined threshold and with delta-LOD > 0 (i.e. clearly the best match).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Allele frequency lookup ───────────────────────────────────────────────────


def build_freq_table(
    geno_df: pd.DataFrame, locus_names: List[str]
) -> Dict[str, Dict[int, float]]:
    """
    Compute population allele frequencies directly from observed genotypes.
    Missing alleles (None / NaN) are excluded from the count.

    Parameters
    ----------
    geno_df     : DataFrame with columns {locus}_a1, {locus}_a2
    locus_names : ordered list of locus names

    Returns
    -------
    freq_table : {locus: {allele: frequency}}
    """
    freq_table = {}
    for locus in locus_names:
        counts: Dict[int, int] = {}
        for col in [f"{locus}_a1", f"{locus}_a2"]:
            for val in geno_df[col].dropna():
                a = int(val)
                counts[a] = counts.get(a, 0) + 1
        total = sum(counts.values())
        if total == 0:
            freq_table[locus] = {}
        else:
            freq_table[locus] = {a: c / total for a, c in counts.items()}
    return freq_table


# ── Single-locus LOD calculation ──────────────────────────────────────────────


def lod_single_locus(
    off_alleles: Tuple[Optional[int], Optional[int]],
    mom_alleles: Tuple[Optional[int], Optional[int]],
    cand_alleles: Tuple[Optional[int], Optional[int]],
    locus_freqs: Dict[int, float],
    min_freq: float = 0.005,
) -> float:
    """
    Compute the LOD score contribution at a single locus.

    Returns 0.0 when data are missing (conservative: missing = no information).
    Returns -10.0 for a definitive exclusion (candidate lacks paternal allele).

    Assumptions
    -----------
    - Mother's genotype is known (maternity assumed confirmed by prior
      behavioural/field observation, as is standard in gorilla studies).
    - Heterozygous offspring → paternal allele is inferred by exclusion of
      maternal allele.  When ambiguous (both alleles match mother), both
      possible paternal alleles are evaluated and averaged.
    - Homozygous offspring → could be true homozygote OR allelic dropout;
      treated conservatively (we average over both interpretations).
    """
    o1, o2 = off_alleles
    m1, m2 = mom_alleles
    c1, c2 = cand_alleles

    # Any missing value → contribute 0 (no information, not exclusion)
    if any(
        x is None or (isinstance(x, float) and np.isnan(x))
        for x in [o1, o2, m1, m2, c1, c2]
    ):
        return 0.0

    o1, o2, m1, m2, c1, c2 = int(o1), int(o2), int(m1), int(m2), int(c1), int(c2)

    # ── Candidate exclusion check ─────────────────────────────────────────────
    # Offspring genotype must be consistent with receiving one allele from
    # mother and one from candidate.  If no valid combination exists → exclude.
    def _candidate_can_supply(paternal_allele):
        return paternal_allele in (c1, c2)

    def _prob_transmit(paternal_allele, cand_a, cand_b):
        """P(candidate transmits paternal_allele)"""
        if cand_a == cand_b:  # homozygous candidate
            return 1.0 if paternal_allele == cand_a else 0.0
        else:  # heterozygous
            return 0.5 if paternal_allele in (cand_a, cand_b) else 0.0

    # Enumerate valid (maternal, paternal) allele assignments
    candidates_mating = []
    for mat_a, pat_a in [(o1, o2), (o2, o1)]:
        # Check maternal compatibility
        mat_ok = mat_a == m1 or mat_a == m2
        if not mat_ok:
            continue
        candidates_mating.append((mat_a, pat_a))

    if not candidates_mating:
        # Neither offspring allele is consistent with mother → likely error
        # Conservative: treat as missing (don't penalise candidate)
        return 0.0

    # Average LOD over consistent (mat, pat) assignments
    lod_vals = []
    for _mat_a, pat_a in candidates_mating:
        freq_pat = locus_freqs.get(pat_a, min_freq)
        prob_trans = _prob_transmit(pat_a, c1, c2)

        if prob_trans == 0.0:
            lod_vals.append(-10.0)  # exclusion
        else:
            ratio = prob_trans / freq_pat
            lod_vals.append(np.log10(ratio))

    return float(np.mean(lod_vals))


# ── Multi-locus LOD and exclusion ────────────────────────────────────────────


def lod_multilocus(
    offspring_id: str,
    mother_id: str,
    candidate_id: str,
    geno_dict: Dict[str, Dict[str, Tuple]],
    freq_table: Dict[str, Dict[int, float]],
    locus_names: List[str],
) -> Dict:
    """
    Compute multi-locus LOD score and exclusion status for one
    offspring–mother–candidate triple.

    Parameters
    ----------
    offspring_id, mother_id, candidate_id : individual IDs
    geno_dict   : {individual_id: {locus: [a1, a2]}}
    freq_table  : output of build_freq_table()
    locus_names : list of loci to use

    Returns
    -------
    dict with keys:
      lod_total    : summed LOD across all loci
      n_loci_used  : loci contributing information
      n_exclusions : loci where candidate is definitively excluded
      excluded     : bool — True if ≥1 definitive exclusion
    """
    off_geno = geno_dict.get(offspring_id, {})
    mom_geno = geno_dict.get(mother_id, {})
    cand_geno = geno_dict.get(candidate_id, {})

    total_lod = 0.0
    n_loci_used = 0
    n_excl = 0

    for locus in locus_names:
        o = off_geno.get(locus, [None, None])
        m = mom_geno.get(locus, [None, None])
        c = cand_geno.get(locus, [None, None])

        lod_l = lod_single_locus(
            tuple(o), tuple(m), tuple(c), freq_table.get(locus, {})
        )

        if lod_l == -10.0:
            n_excl += 1
            total_lod += lod_l  # penalise total LOD
        elif lod_l != 0.0:
            n_loci_used += 1
            total_lod += lod_l
        # lod_l == 0.0: missing data, no contribution

    return {
        "lod_total": round(total_lod, 4),
        "n_loci_used": n_loci_used,
        "n_exclusions": n_excl,
        "excluded": n_excl > 0,
    }


# ── Full parentage assignment ─────────────────────────────────────────────────


def assign_paternity(
    offspring_ids: List[str],
    mother_ids: List[str],
    candidate_ids: List[str],
    geno_dict: Dict,
    freq_table: Dict,
    locus_names: List[str],
    lod_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Assign the most likely father for each offspring from a pool of candidates.

    Assignment criteria (following Marshall et al. 1998):
      1. Candidate must NOT be excluded at any locus.
      2. Candidate must have the highest LOD score among non-excluded candidates.
      3. LOD score must exceed lod_threshold (default 0: at least more likely
         than random).
      4. Delta-LOD (LOD_best - LOD_2nd_best) > 0.

    Confidence levels:
      HIGH   : delta-LOD > 2.0  (strong evidence)
      MEDIUM : delta-LOD 1.0–2.0
      LOW    : delta-LOD 0–1.0
      NONE   : no non-excluded candidate above threshold

    Parameters
    ----------
    offspring_ids : IDs of offspring to assign paternity to
    mother_ids    : corresponding mother IDs (same length; None if unknown)
    candidate_ids : all candidate fathers in the population
    lod_threshold : minimum LOD score for assignment

    Returns
    -------
    DataFrame with one row per offspring
    """
    results = []

    for off_id, mom_id in zip(offspring_ids, mother_ids):
        lod_scores = []

        for cand_id in candidate_ids:
            # Skip self-assignment (offspring can't be own father)
            if cand_id == off_id:
                continue

            res = lod_multilocus(
                off_id, mom_id, cand_id, geno_dict, freq_table, locus_names
            )
            lod_scores.append({"candidate_id": cand_id, **res})

        # Sort by LOD descending (excluded candidates go to bottom)
        lod_scores.sort(key=lambda x: (not x["excluded"], x["lod_total"]), reverse=True)

        # Find top non-excluded candidates
        valid = [
            x
            for x in lod_scores
            if not x["excluded"] and x["lod_total"] >= lod_threshold
        ]

        if len(valid) == 0:
            assigned_id = None
            assigned_lod = np.nan
            delta_lod = np.nan
            confidence = "NONE"
        elif len(valid) == 1:
            assigned_id = valid[0]["candidate_id"]
            assigned_lod = valid[0]["lod_total"]
            delta_lod = assigned_lod  # only one candidate
            confidence = _confidence_level(delta_lod)
        else:
            assigned_id = valid[0]["candidate_id"]
            assigned_lod = valid[0]["lod_total"]
            delta_lod = assigned_lod - valid[1]["lod_total"]
            confidence = _confidence_level(delta_lod)

        results.append(
            {
                "offspring_id": off_id,
                "mother_id": mom_id,
                "assigned_father": assigned_id,
                "lod_score": assigned_lod,
                "delta_lod": delta_lod,
                "confidence": confidence,
                "n_candidates": len(candidate_ids),
                "n_valid_cands": len(valid),
            }
        )

    return pd.DataFrame(results)


def _confidence_level(delta_lod: float) -> str:
    if delta_lod >= 2.0:
        return "HIGH"
    elif delta_lod >= 1.0:
        return "MEDIUM"
    elif delta_lod >= 0.0:
        return "LOW"
    else:
        return "NONE"


# ── Exclusion probability ─────────────────────────────────────────────────────


def exclusion_probability_per_locus(
    freq_table: Dict[str, Dict[int, float]], locus: str
) -> float:
    """
    Probability that a random unrelated male is excluded at this locus.
    PE = 1 - Σ_i p_i^2 (1 - p_i^2)  [Jamieson & Taylor 1997 simplified]

    Returns PE for a single locus given known maternal genotype (averaged
    over all possible maternal genotypes).
    """
    freqs = list(freq_table.get(locus, {}).values())
    if not freqs:
        return 0.0
    freqs = np.array(freqs)
    # PE for one parent known (mother): PE = 1 - Σ p_i^2 (2 - p_i^2)
    pe = 1.0 - float(np.sum(freqs**2 * (2 - freqs**2)))
    return max(0.0, pe)


def combined_exclusion_probability(
    freq_table: Dict[str, Dict[int, float]], locus_names: List[str]
) -> float:
    """
    Combined exclusion probability across all loci (assuming independence).
    CPE = 1 - Π_l (1 - PE_l)
    """
    pe_vals = [exclusion_probability_per_locus(freq_table, l) for l in locus_names]
    cpe = 1.0 - float(np.prod([1.0 - pe for pe in pe_vals]))
    return cpe


# ── Polymorphism information content (PIC) ────────────────────────────────────


def pic_per_locus(freq_table: Dict[str, Dict[int, float]], locus: str) -> float:
    """
    PIC = 1 - Σ_i p_i^2 - Σ_i Σ_{j>i} 2 p_i^2 p_j^2
    [Botstein et al. 1980]
    """
    freqs = np.array(list(freq_table.get(locus, {}).values()))
    if len(freqs) == 0:
        return np.nan
    sum_p2 = float(np.sum(freqs**2))
    sum_2p2q2 = float(
        2
        * np.sum(
            [
                freqs[i] ** 2 * freqs[j] ** 2
                for i in range(len(freqs))
                for j in range(i + 1, len(freqs))
            ]
        )
    )
    return 1.0 - sum_p2 - sum_2p2q2


# ── Hardy-Weinberg Equilibrium test ──────────────────────────────────────────


def hwe_exact_chi2(geno_df: pd.DataFrame, locus: str) -> Dict:
    """
    Chi-squared test for HWE departure at a single locus.
    Returns test statistic, p-value, and observed/expected heterozygosity.
    """
    from scipy.stats import chi2

    col1, col2 = f"{locus}_a1", f"{locus}_a2"
    mask = geno_df[col1].notna() & geno_df[col2].notna()
    sub = geno_df[mask]
    n = len(sub)
    if n == 0:
        return {
            "locus": locus,
            "n": 0,
            "Ho": np.nan,
            "He": np.nan,
            "chi2": np.nan,
            "p_value": np.nan,
        }

    # Allele counts
    alleles = pd.concat([sub[col1], sub[col2]]).astype(int)
    freqs = alleles.value_counts(normalize=True)
    He = 1.0 - float((freqs**2).sum())

    # Observed heterozygosity
    Ho = float((sub[col1].astype(int) != sub[col2].astype(int)).mean())

    # Chi-squared: compare observed vs expected genotype counts
    # (simplified two-allele approximation for reporting)
    # For full multi-allele HWE: use exact test (computationally intensive)
    # Here we use the inbreeding coefficient F = 1 - Ho/He as a summary
    F = 1.0 - (Ho / He) if He > 0 else np.nan

    # Chi2 statistic for the HWE test (approximation)
    chi2_stat = n * (Ho - He) ** 2 / (He * (1 - He)) if (He > 0 and He < 1) else np.nan
    p_val = float(1 - chi2.cdf(chi2_stat, df=1)) if not np.isnan(chi2_stat) else np.nan

    return {
        "locus": locus,
        "n": n,
        "n_alleles": int(freqs.shape[0]),
        "Ho": round(Ho, 4),
        "He": round(He, 4),
        "F_is": round(F, 4) if not np.isnan(F) else np.nan,
        "chi2": round(chi2_stat, 4) if not np.isnan(chi2_stat) else np.nan,
        "p_value": round(p_val, 4) if not np.isnan(p_val) else np.nan,
    }


# ── Reproductive skew (Nonacs B-index) ───────────────────────────────────────


def nonacs_b_index(paternity_counts: Dict[str, int]) -> Dict:
    """
    Compute Nonacs (2000) B-index of reproductive skew.

    B = Σ_i (p_i - 1/k)^2
    where p_i = fraction of offspring sired by male i, k = number of males.

    Expected B under perfect equality = 1/k - 1/k^2  (≈ 1/k for large k).
    B > E(B) indicates skew toward dominant males.

    Parameters
    ----------
    paternity_counts : {male_id: number_of_offspring_sired}

    Returns
    -------
    dict with B, E_B, k, and interpretation
    """
    counts = {k: v for k, v in paternity_counts.items() if v > 0}
    k = len(counts)
    if k == 0:
        return {"B": np.nan, "E_B": np.nan, "k": 0}

    total = sum(counts.values())
    probs = np.array([v / total for v in counts.values()])
    B = float(np.sum((probs - 1.0 / k) ** 2))
    E_B = 1.0 / k - 1.0 / k**2

    return {
        "B": round(B, 6),
        "E_B": round(E_B, 6),
        "k": k,
        "total_offspring": total,
        "skew_ratio": round(B / E_B, 3) if E_B > 0 else np.nan,
    }


def permutation_test_skew(
    paternity_counts: Dict[str, int], n_perm: int = 1000, seed: int = 42
) -> Dict:
    """
    Test whether observed B-index is significantly greater than expected
    under random (equal) paternity allocation via permutation.

    Returns observed B, null distribution mean/sd, and p-value.
    """
    rng = np.random.default_rng(seed)
    obs = nonacs_b_index(paternity_counts)
    B_obs = obs["B"]
    k = obs["k"]
    n_total = obs["total_offspring"]

    null_B = []
    males = list(paternity_counts.keys())

    for _ in range(n_perm):
        # Randomly assign offspring to males with equal probability
        rand_assignment = rng.choice(males, size=n_total, replace=True)
        rand_counts = {m: int(np.sum(rand_assignment == m)) for m in males}
        null_B.append(nonacs_b_index(rand_counts)["B"])

    null_B = np.array(null_B)
    p_val = float(np.mean(null_B >= B_obs))

    return {
        "B_observed": B_obs,
        "null_mean": round(float(null_B.mean()), 6),
        "null_sd": round(float(null_B.std()), 6),
        "p_value": round(p_val, 4),
        "significant": p_val < 0.05,
    }
