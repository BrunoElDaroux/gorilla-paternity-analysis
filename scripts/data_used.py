"""
data_generator.py
=================
Generates a synthetic but biologically realistic mountain gorilla
microsatellite dataset for non-invasive parentage and kinship analysis.

Population parameters modelled on:
  - Bradley et al. (2004) Am J Primatol 62:1-14  [Virunga gorillas]
  - Nsubuga et al. (2010) Mol Ecol Resour 10:397-399 [gorilla loci]
  - Robbins et al. (2007) Behav Ecol Sociobiol 61:1219-1228 [skew]

Run from project root:
    python scripts/data_generator.py
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Microsatellite loci ────────────────────────────────────────────────────────
# 15 loci commonly used in gorilla studies.
# Allele size ranges (repeat units) approximated from published panels.
# Dirichlet-distributed frequencies are generated to mimic real variation.

LOCI_DEF = {
    "D2S1338": range(19, 32),
    "D3S1358": range(14, 21),
    "D5S818": range(7, 16),
    "D7S820": range(6, 15),
    "D8S1179": range(10, 19),
    "D13S317": range(8, 15),
    "D16S539": range(9, 16),
    "D18S51": range(12, 24),
    "D19S433": range(12, 18),
    "D21S11": range(27, 38),
    "vWA": range(14, 22),
    "FGA": range(19, 31),
    "CSF1PO": range(9, 16),
    "TPOX": range(8, 14),
    "TH01": range(5, 12),
}

LOCUS_NAMES = list(LOCI_DEF.keys())

# Generate population-level allele frequencies via Dirichlet distribution.
# concentration α ≈ 1.5 produces moderately skewed spectra (realistic for
# gorillas with intermediate effective population sizes).
LOCUS_ALLELES = {}
LOCUS_FREQS = {}

for locus, allele_range in LOCI_DEF.items():
    alleles = list(allele_range)
    n = len(alleles)
    freqs = np.random.dirichlet(np.ones(n) * 1.5)
    LOCUS_ALLELES[locus] = alleles
    LOCUS_FREQS[locus] = dict(zip(alleles, freqs))

# ── Social group definitions ───────────────────────────────────────────────────
GROUPS = [
    {"id": "G1", "name": "Susa", "location": "Karisimbi"},
    {"id": "G2", "name": "Amahoro", "location": "Karisimbi"},
    {"id": "G3", "name": "Umubano", "location": "Bisoke"},
    {"id": "G4", "name": "Sabyinyo", "location": "Sabyinyo"},
    {"id": "G5", "name": "Hirwa", "location": "Muhabura"},
    {"id": "G6", "name": "Kwitonda", "location": "Muhabura"},
]

# ── Reproductive skew parameters ───────────────────────────────────────────────
# Robbins et al. (2007): dominant males sire ~59-67% of offspring in
# multi-male groups. We model the rest as subordinate or extra-group.
PROB_DOM = 0.62  # dominant silverback
PROB_SUB = 0.28  # subordinate male(s) within group
PROB_EXT = 0.10  # extra-group / unsampled male

# ── Non-invasive DNA error parameters ─────────────────────────────────────────
# Modelled after Arandjelovic et al. (2009) and Nsubuga et al. (2010).
# These apply ONLY to simulated genotyping of field-collected samples.
ADO_RATE = 0.08  # allelic dropout per allele per locus
FALSE_ALLELE = 0.02  # false allele rate per locus
LOCUS_FAIL = 0.07  # complete locus failure rate

# ── Counter for unique IDs ────────────────────────────────────────────────────
_ID_COUNTER = {"n": 1}


def _new_id(prefix: str) -> str:
    uid = f"{prefix}{_ID_COUNTER['n']:03d}"
    _ID_COUNTER["n"] += 1
    return uid


# ── Genotype functions ─────────────────────────────────────────────────────────


def draw_from_population(locus: str) -> list:
    """Draw two alleles independently from population frequencies."""
    alleles = LOCUS_ALLELES[locus]
    freqs = list(LOCUS_FREQS[locus].values())
    a1 = np.random.choice(alleles, p=freqs)
    a2 = np.random.choice(alleles, p=freqs)
    return [int(a1), int(a2)]


def mendelian_draw(parent1_geno: dict, parent2_geno: dict, locus: str) -> list:
    """
    Transmit one allele from each parent (Mendelian inheritance).
    Includes stepwise microsatellite mutation at rate 10^-3 per allele.
    """
    p1 = parent1_geno.get(locus, [None, None])
    p2 = parent2_geno.get(locus, [None, None])

    if None in p1 or None in p2:
        return draw_from_population(locus)

    a1 = random.choice(p1)
    a2 = random.choice(p2)

    alleles = LOCUS_ALLELES[locus]
    lo, hi = min(alleles), max(alleles)

    # Stepwise mutation (SMM)
    if random.random() < 1e-3:
        a1 = max(lo, min(hi, a1 + random.choice([-1, 1])))
    if random.random() < 1e-3:
        a2 = max(lo, min(hi, a2 + random.choice([-1, 1])))

    return [int(a1), int(a2)]


def generate_true_genotype(parent1: dict = None, parent2: dict = None) -> dict:
    """
    Generate a complete multi-locus true genotype.
    If parents provided → Mendelian; otherwise → draw from population.
    """
    geno = {}
    for locus in LOCUS_NAMES:
        if parent1 and parent2:
            geno[locus] = mendelian_draw(parent1, parent2, locus)
        else:
            geno[locus] = draw_from_population(locus)
    return geno


def apply_noninvasive_errors(true_geno: dict) -> dict:
    """
    Simulate genotyping errors typical of non-invasive (faecal/hair) DNA:
      - Locus failure:  both alleles become None
      - Allelic dropout: one allele is replaced by the other (apparent homozygote)
      - False allele:   one allele changes to a neighbouring repeat unit
    """
    obs = {}
    for locus, (a1, a2) in true_geno.items():
        # Complete locus failure
        if random.random() < LOCUS_FAIL:
            obs[locus] = [None, None]
            continue

        # Allelic dropout on each allele independently
        if random.random() < ADO_RATE:
            a1 = a2  # apparent homozygote
        if random.random() < ADO_RATE:
            a2 = a1

        # False allele (stutter / contamination)
        if random.random() < FALSE_ALLELE:
            alleles = LOCUS_ALLELES[locus]
            a1 = max(min(alleles), min(max(alleles), a1 + random.choice([-1, 1])))

        obs[locus] = [int(a1), int(a2)]
    return obs


# ── Build population ───────────────────────────────────────────────────────────

individuals = []  # list of dicts (metadata)
true_genotypes = {}  # {id: {locus: [a1, a2]}}  (error-free)
obs_genotypes = {}  # {id: {locus: [a1|None, a2|None]}}  (with errors)

for grp in GROUPS:
    gid = grp["id"]

    # ── Dominant silverback ────────────────────────────────────────────────────
    dom_id = _new_id("SB")
    dom_true = generate_true_genotype()
    true_genotypes[dom_id] = dom_true
    obs_genotypes[dom_id] = apply_noninvasive_errors(dom_true)
    individuals.append(
        {
            "individual_id": dom_id,
            "name": f"{grp['name']}_Dom",
            "sex": "M",
            "age_class": "silverback",
            "group_id": gid,
            "dominance_rank": 1,
            "mother_id": None,
            "true_father_id": None,
        }
    )

    # ── Subordinate males (0–2) ───────────────────────────────────────────────
    n_sub = random.randint(0, 2)
    sub_ids = []
    for s in range(n_sub):
        sid = _new_id("SB")
        sub_true = generate_true_genotype()
        true_genotypes[sid] = sub_true
        obs_genotypes[sid] = apply_noninvasive_errors(sub_true)
        individuals.append(
            {
                "individual_id": sid,
                "name": f"{grp['name']}_Sub{s+1}",
                "sex": "M",
                "age_class": "silverback",
                "group_id": gid,
                "dominance_rank": s + 2,
                "mother_id": None,
                "true_father_id": None,
            }
        )
        sub_ids.append(sid)

    # ── Adult females (3–6) ───────────────────────────────────────────────────
    n_fem = random.randint(3, 6)
    fem_ids = []
    fem_true_genos = {}
    for f in range(n_fem):
        fid = _new_id("AF")
        fem_true = generate_true_genotype()
        fem_true_genos[fid] = fem_true
        true_genotypes[fid] = fem_true
        obs_genotypes[fid] = apply_noninvasive_errors(fem_true)
        individuals.append(
            {
                "individual_id": fid,
                "name": f"{grp['name']}_AF{f+1}",
                "sex": "F",
                "age_class": "adult",
                "group_id": gid,
                "dominance_rank": None,
                "mother_id": None,
                "true_father_id": None,
            }
        )
        fem_ids.append(fid)

    # ── Offspring (2–5) ───────────────────────────────────────────────────────
    n_off = random.randint(2, 5)
    for o in range(n_off):
        # Assign true father
        r = random.random()
        if r < PROB_DOM:
            true_father_id = dom_id
            father_true = true_genotypes[dom_id]
        elif r < PROB_DOM + PROB_SUB and sub_ids:
            true_father_id = random.choice(sub_ids)
            father_true = true_genotypes[true_father_id]
        else:
            true_father_id = "UNSAMPLED"
            father_true = generate_true_genotype()  # simulate unsampled male

        true_mother_id = random.choice(fem_ids)
        mother_true = fem_true_genos[true_mother_id]

        oid = _new_id("JUV")
        off_true = generate_true_genotype(mother_true, father_true)
        true_genotypes[oid] = off_true
        obs_genotypes[oid] = apply_noninvasive_errors(off_true)

        individuals.append(
            {
                "individual_id": oid,
                "name": f"{grp['name']}_OFF{o+1}",
                "sex": random.choice(["M", "F"]),
                "age_class": random.choice(["infant", "juvenile"]),
                "group_id": gid,
                "dominance_rank": None,
                "mother_id": true_mother_id,
                "true_father_id": true_father_id,
            }
        )

# ── Assemble DataFrames ────────────────────────────────────────────────────────

ind_df = pd.DataFrame(individuals)

# Observed genotypes → wide format
geno_rows = []
for iid, geno in obs_genotypes.items():
    row = {"individual_id": iid}
    for locus in LOCUS_NAMES:
        alleles = geno.get(locus, [None, None])
        row[f"{locus}_a1"] = alleles[0]
        row[f"{locus}_a2"] = alleles[1]
    geno_rows.append(row)

geno_df = pd.DataFrame(geno_rows)

# True (error-free) genotypes — kept for validation only
true_geno_rows = []
for iid, geno in true_genotypes.items():
    row = {"individual_id": iid}
    for locus in LOCUS_NAMES:
        row[f"{locus}_a1_true"] = geno[locus][0]
        row[f"{locus}_a2_true"] = geno[locus][1]
    true_geno_rows.append(row)

true_geno_df = pd.DataFrame(true_geno_rows)

# Allele frequency table
freq_rows = []
for locus in LOCUS_NAMES:
    for allele, freq in LOCUS_FREQS[locus].items():
        freq_rows.append(
            {"locus": locus, "allele": allele, "frequency": round(freq, 6)}
        )

freq_df = pd.DataFrame(freq_rows)

groups_df = pd.DataFrame(GROUPS)

# ── Save ──────────────────────────────────────────────────────────────────────

ind_df.to_csv(DATA_DIR / "individuals.csv", index=False)
geno_df.to_csv(DATA_DIR / "genotypes.csv", index=False)
true_geno_df.to_csv(DATA_DIR / "genotypes_true.csv", index=False)
freq_df.to_csv(DATA_DIR / "allele_frequencies.csv", index=False)
groups_df.to_csv(DATA_DIR / "groups.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Mountain Gorilla Dataset — Generation Complete")
print("=" * 60)
print(f"  Total individuals : {len(ind_df)}")
print(f"  Groups            : {len(groups_df)}")
print(f"  Loci              : {len(LOCUS_NAMES)}")
print()
print("  Age class breakdown:")
print(ind_df["age_class"].value_counts().to_string(header=False))
print()
print("  Sex breakdown:")
print(ind_df["sex"].value_counts().to_string(header=False))
print()

offspring = ind_df[ind_df["true_father_id"].notna() & (ind_df["true_father_id"] != "")]
n_off = len(offspring)
if n_off:
    n_dom = (
        offspring["true_father_id"]
        == offspring["group_id"].map(
            lambda g: next(
                (
                    i["individual_id"]
                    for i in individuals
                    if i["group_id"] == g and i["dominance_rank"] == 1
                ),
                None,
            )
        )
    ).sum()
    n_extra = (offspring["true_father_id"] == "UNSAMPLED").sum()
    n_sub = n_off - n_extra - n_dom
    print(f"  Offspring paternity (ground truth):")
    print(f"    Dominant male  : {n_dom}  ({100*n_dom/n_off:.1f}%)")
    print(f"    Subordinate(s) : {n_sub}  ({100*n_sub/n_off:.1f}%)")
    print(f"    Extra-group    : {n_extra}  ({100*n_extra/n_off:.1f}%)")
print()
print(f"  Files saved to: {DATA_DIR}")
print("=" * 60)
