"""
Generate Synthetic Student Records for ML Training
===================================================
Uses the REAL scraped ACPC cutoff data (acpc_cutoffs.csv) and generates
synthetic student-level records with admission outcomes for model training.

Each cutoff row defines a closing rank threshold.
We generate students who were admitted (rank <= closing_rank)
and students who were rejected (rank > closing_rank).

This also adds computed features:
  - gujcet_score (estimated from rank)
  - twelfth_pcm_percent (estimated from rank)
  - college_tier (1/2/3 based on college selectivity)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

np.random.seed(42)

DATA_DIR = Path(__file__).parent
CUTOFF_FILE = DATA_DIR / "acpc_cutoffs.csv"
OUTPUT_FILE = DATA_DIR / "student_data.csv"

# ── Tier classification based on average closing rank ─────────────────────────
TIER1_KEYWORDS = [
    "daiict", "nirma", "pdeu", "svnit", "ldce", "l.d.", "dhirubhai",
    "iit", "nit", "iiit",
]
TIER2_KEYWORDS = [
    "bvm", "vgec", "gcet", "msu", "faculty of tech",
    "marwadi", "silver oak", "charotar", "birla",
    "institute of tech", "vishwakarma",
]


def classify_tier(college_name: str) -> int:
    """Classify college into tier 1/2/3 based on name keywords."""
    name_lower = college_name.lower()
    if any(kw in name_lower for kw in TIER1_KEYWORDS):
        return 1
    if any(kw in name_lower for kw in TIER2_KEYWORDS):
        return 2
    return 3


def rank_to_gujcet(rank: int) -> int:
    """Estimate GUJCET score (0-120) from merit rank."""
    score = max(5, min(120, int(120 - (rank / 60000) * 110)))
    return int(np.clip(score + np.random.randint(-3, 4), 0, 120))


def rank_to_pcm(rank: int) -> float:
    """Estimate 12th PCM percentage from merit rank."""
    pct = max(40.0, min(99.5, 99.0 - (rank / 60000) * 55))
    return round(np.clip(pct + np.random.uniform(-3, 3), 35.0, 99.9), 1)


def normalize_branch(branch: str) -> str:
    """Clean up branch names."""
    branch = str(branch).strip()
    b = branch.lower()
    if "computer" in b and ("science" in b or "cse" in b):
        return "Computer Science"
    if "computer" in b:
        return "Computer Engineering"
    if "information" in b and "tech" in b:
        return "Information Technology"
    if "electronics" in b or "ec" == b:
        return "Electronics & Communication"
    if "electrical" in b:
        return "Electrical Engineering"
    if "mechani" in b:
        return "Mechanical Engineering"
    if "civil" in b:
        return "Civil Engineering"
    if "chemical" in b:
        return "Chemical Engineering"
    if "auto" in b:
        return "Automobile Engineering"
    if "bio" in b:
        return "Biomedical Engineering"
    if "instrument" in b:
        return "Instrumentation"
    if "environ" in b:
        return "Environmental Engineering"
    if "food" in b:
        return "Food Technology"
    if "plastic" in b or "rubber" in b or "polymer" in b:
        return "Plastic/Polymer Engineering"
    if "textile" in b:
        return "Textile Engineering"
    if "mining" in b:
        return "Mining Engineering"
    return branch.title()


def normalize_category(cat: str) -> str:
    """Ensure category is properly labeled."""
    c = str(cat).strip().upper()
    mapping = {
        "GENERAL": "General", "GEN": "General", "OPEN": "General",
        "OBC": "OBC", "SEBC": "OBC",
        "SC": "SC", "ST": "ST", "EWS": "EWS",
        "TFWS": "TFWS", "PH": "PH",
    }
    return mapping.get(c, cat)


def main():
    print("=" * 60)
    print("  Generating Student Training Data from ACPC Cutoffs")
    print("=" * 60)

    if not CUTOFF_FILE.exists():
        print(f"\nERROR: {CUTOFF_FILE} not found!")
        print("Run 'python data/scrape_acpc.py' first.")
        sys.exit(1)

    df = pd.read_csv(CUTOFF_FILE)
    print(f"\nLoaded {len(df)} cutoff records")

    # Clean columns
    df["college"] = df["college"].fillna("Unknown").astype(str).str.strip()
    df["branch"] = df["branch"].fillna("Unknown").astype(str)
    df["branch"] = df["branch"].apply(normalize_branch)
    df["category"] = df["category"].fillna("General").apply(normalize_category)
    df["college_tier"] = df["college"].apply(classify_tier)

    # Remove rows with bad data
    df = df.dropna(subset=["closing_rank"])
    df = df[df["closing_rank"] > 0]

    # Since scraped data only has "General" category,
    # expand with synthetic category variations
    categories = ["General", "OBC", "SC", "ST", "EWS"]
    cat_multipliers = {
        "General": 1.0, "EWS": 1.15, "OBC": 1.30, "SC": 1.80, "ST": 2.20
    }

    expanded_rows = []
    for _, row in df.iterrows():
        for cat in categories:
            mult = cat_multipliers[cat]
            new_closing = int(row["closing_rank"] * mult * np.random.uniform(0.95, 1.05))
            new_opening = int(row["opening_rank"] * mult * np.random.uniform(0.95, 1.05)) if pd.notna(row["opening_rank"]) else max(1, new_closing // 3)
            expanded_rows.append({
                "college": row["college"],
                "branch": row["branch"],
                "category": cat,
                "year": row["year"],
                "college_tier": row["college_tier"],
                "opening_rank": new_opening,
                "closing_rank": new_closing,
            })

    cutoff_expanded = pd.DataFrame(expanded_rows)
    print(f"Expanded to {len(cutoff_expanded)} cutoff rows (with categories)")

    # Generate student records
    student_records = []
    for _, row in cutoff_expanded.iterrows():
        closing = int(row["closing_rank"])
        opening = int(row["opening_rank"]) if pd.notna(row["opening_rank"]) else max(1, closing // 3)

        # Fix: ensure opening <= closing (PDF parsing sometimes swaps them)
        if opening > closing:
            opening, closing = closing, opening
        if closing <= 0:
            continue

        # Generate 2-4 admitted students
        n_admitted = np.random.randint(2, 5)
        for _ in range(n_admitted):
            low = max(1, opening - 500)
            high = closing + 1
            if low >= high:
                low = max(1, high - 100)
            rank = np.random.randint(low, max(low + 1, high))
            student_records.append({
                "college": row["college"],
                "branch": row["branch"],
                "category": row["category"],
                "year": row["year"],
                "college_tier": row["college_tier"],
                "student_rank": rank,
                "gujcet_score": rank_to_gujcet(rank),
                "twelfth_pcm_percent": rank_to_pcm(rank),
                "opening_rank": opening,
                "closing_rank": closing,
                "admitted": 1,
            })

        # Generate 2-4 rejected students
        n_rejected = np.random.randint(2, 5)
        for _ in range(n_rejected):
            rank = np.random.randint(closing + 1, closing + 8000)
            student_records.append({
                "college": row["college"],
                "branch": row["branch"],
                "category": row["category"],
                "year": row["year"],
                "college_tier": row["college_tier"],
                "student_rank": rank,
                "gujcet_score": rank_to_gujcet(rank),
                "twelfth_pcm_percent": rank_to_pcm(rank),
                "opening_rank": opening,
                "closing_rank": closing,
                "admitted": 0,
            })

    student_df = pd.DataFrame(student_records)

    # Save
    student_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total student records: {len(student_df):,}")
    print(f"  Admitted: {student_df['admitted'].sum():,} ({student_df['admitted'].mean():.1%})")
    print(f"  Rejected: {(1-student_df['admitted']).sum():.0f} ({1-student_df['admitted'].mean():.1%})")
    print(f"  Unique colleges: {student_df['college'].nunique()}")
    print(f"  Unique branches: {student_df['branch'].nunique()}")
    print(f"  Categories: {sorted(student_df['category'].unique())}")
    print(f"  Years: {sorted(student_df['year'].unique())}")
    print(f"  Tier distribution: {student_df['college_tier'].value_counts().to_dict()}")
    print(f"\n  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
