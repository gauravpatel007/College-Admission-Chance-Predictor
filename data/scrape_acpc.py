"""
ACPC Gujarat Engineering Cutoff Data Scraper (Option A)
=======================================================
Scrapes cutoff PDF links from acpc.gujarat.gov.in/be-btech-archives,
downloads the Institute-wise cutoff PDFs for 4 years (2020-21 to 2023-24),
and parses them with pdfplumber into a clean CSV.

Source: https://acpc.gujarat.gov.in/be-btech-archives
"""

import os
import re
import sys
import time
import requests
import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL = "https://acpc.gujarat.gov.in"
ARCHIVES_URL = f"{BASE_URL}/be-btech-archives"

DATA_DIR = Path(__file__).parent
PDF_DIR = DATA_DIR / "pdfs"
PDF_DIR.mkdir(exist_ok=True)

# PDFs we want to download (institute-wise cutoffs give college+branch+category+rank)
# Manually curated list from the archives page - these are the institute-wise cutoff PDFs
TARGET_PDFS = {
    "2024-25": f"{BASE_URL}/assets/uploads/media-uploader/closure-2024-25-degree-engineering1742967781.pdf",
    "2023-24": f"{BASE_URL}/assets/uploads/media-uploader/cut-off-marks-for-the-year-2023-24-institute-wise1742982441.pdf",
    "2022-23": f"{BASE_URL}/assets/uploads/media-uploader/institute-wise-engineering-closure-20231683613447.pdf",
    "2021-22": f"{BASE_URL}/assets/uploads/media-uploader/institute-wise-engineering-closure-20211683613976.pdf",
    "2020-21": f"{BASE_URL}/assets/uploads/media-uploader/cut-off-marks-for-the-year-2020-21-institutewise1652780645.pdf",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ── Step 1: Download PDFs ─────────────────────────────────────────────────────
def download_pdf(url: str, year: str) -> Path:
    """Download a PDF and save it locally. Returns the file path."""
    filename = f"acpc_cutoff_{year.replace('-', '_')}.pdf"
    filepath = PDF_DIR / filename

    if filepath.exists():
        print(f"  ✓ Already downloaded: {filename}")
        return filepath

    print(f"  ↓ Downloading {year} cutoff from: {url[:80]}...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  ✓ Saved: {filename} ({filepath.stat().st_size / 1024:.0f} KB)")
        time.sleep(1)  # Be polite to the server
        return filepath
    except Exception as e:
        print(f"  ✗ Failed to download {year}: {e}")
        return None


# ── Step 2: Parse PDF tables ──────────────────────────────────────────────────
def parse_cutoff_pdf(filepath: Path, year: str) -> pd.DataFrame:
    """
    Parse an ACPC institute-wise cutoff PDF.
    These PDFs typically have tables with columns like:
    Institute Code | Institute Name | Program | Category | Opening Rank | Closing Rank
    
    The exact format varies by year, so we use heuristic parsing.
    """
    if filepath is None:
        return pd.DataFrame()

    print(f"\n📄 Parsing {filepath.name}...")
    all_rows = []

    try:
        with pdfplumber.open(filepath) as pdf:
            print(f"   Pages: {len(pdf.pages)}")

            current_institute = ""
            current_program = ""

            for page_num, page in enumerate(pdf.pages):
                # Extract tables from the page
                tables = page.extract_tables()

                if not tables:
                    # Try extracting text line by line
                    text = page.extract_text()
                    if text:
                        lines = text.strip().split("\n")
                        for line in lines:
                            row_data = parse_text_line(
                                line, current_institute, current_program, year
                            )
                            if row_data:
                                if row_data.get("_institute"):
                                    current_institute = row_data["_institute"]
                                if row_data.get("_program"):
                                    current_program = row_data["_program"]
                                if row_data.get("closing_rank"):
                                    all_rows.append(row_data)
                    continue

                for table in tables:
                    if not table:
                        continue

                    for row in table:
                        if not row or all(cell is None or str(cell).strip() == "" for cell in row):
                            continue

                        # Clean cells
                        cells = [str(c).strip() if c else "" for c in row]

                        # Skip header rows
                        if is_header_row(cells):
                            continue

                        # Try to extract data from row
                        parsed = parse_table_row(
                            cells, current_institute, current_program, year
                        )
                        if parsed:
                            if parsed.get("_institute"):
                                current_institute = parsed["_institute"]
                            if parsed.get("_program"):
                                current_program = parsed["_program"]
                            if parsed.get("closing_rank"):
                                all_rows.append(parsed)

            if page_num % 20 == 0:
                print(f"   Processed page {page_num + 1}/{len(pdf.pages)}...")

    except Exception as e:
        print(f"   ✗ Error parsing {filepath.name}: {e}")
        import traceback
        traceback.print_exc()

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    if not df.empty:
        # Clean up internal columns
        for col in ["_institute", "_program"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Remove rows where closing_rank is not numeric
        df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce")
        df["opening_rank"] = pd.to_numeric(df["opening_rank"], errors="coerce")
        df.dropna(subset=["closing_rank"], inplace=True)

    print(f"   ✓ Extracted {len(df)} records from {filepath.name}")
    return df


def is_header_row(cells: list) -> bool:
    """Check if a row is a header row."""
    header_keywords = [
        "institute", "program", "category", "opening", "closing",
        "rank", "sr.", "sr no", "code", "merit", "branch", "course"
    ]
    joined = " ".join(cells).lower()
    return sum(1 for kw in header_keywords if kw in joined) >= 2


def parse_table_row(
    cells: list, current_institute: str, current_program: str, year: str
) -> dict:
    """
    Parse a single table row. ACPC PDFs often have these patterns:
    
    Pattern 1: [Code, Institute, Program, Category, OpenRank, CloseRank]
    Pattern 2: [Institute, Program, Category, OpenRank, CloseRank]
    Pattern 3: [Category, OpenRank, CloseRank] (when institute/program span rows)
    """
    result = {"year": year, "_institute": None, "_program": None}

    # Filter out empty cells
    non_empty = [c for c in cells if c.strip()]

    if len(non_empty) < 2:
        return None

    # Try to find rank-like numbers (4-6 digit numbers or dashes for "no admission")
    rank_pattern = re.compile(r"^\d{1,6}$")
    ranks = []
    rank_indices = []
    for i, cell in enumerate(cells):
        clean = cell.replace(",", "").replace(" ", "").strip()
        if rank_pattern.match(clean):
            ranks.append(int(clean))
            rank_indices.append(i)

    # We need at least 2 rank-like numbers (opening and closing)
    if len(ranks) >= 2:
        result["opening_rank"] = ranks[-2]
        result["closing_rank"] = ranks[-1]

        # Everything before ranks is descriptive
        desc_cells = cells[:rank_indices[-2]]
        desc_text = " ".join(c.strip() for c in desc_cells if c.strip())

        # Try to identify category
        categories = [
            "General", "GEN", "OPEN", "OBC", "SEBC",
            "SC", "ST", "EWS", "TFWS", "PH", "NRI", "AI"
        ]
        found_category = None
        for cat in categories:
            for cell in cells:
                if cat.upper() in cell.upper().split():
                    found_category = normalize_category(cat)
                    break
            if found_category:
                break

        # Fallback: check if one of the non-rank cells is a category
        if not found_category:
            for cell in cells:
                norm = normalize_category(cell.strip())
                if norm:
                    found_category = norm
                    break

        result["category"] = found_category or "General"

        # Try to find institute name in the row
        for cell in cells:
            if len(cell) > 10 and not rank_pattern.match(cell.replace(",", "").strip()):
                if any(kw in cell.lower() for kw in [
                    "institute", "college", "university", "engg",
                    "engineering", "tech", "polytechnic", "school"
                ]):
                    result["_institute"] = cell.strip()
                    result["college"] = cell.strip()
                    break

        if "college" not in result:
            result["college"] = current_institute

        # Try to find program/branch name
        branch_keywords = [
            "computer", "information", "civil", "mechanical", "electrical",
            "electronics", "chemical", "automobile", "biomedical",
            "instrumentation", "mining", "textile", "environmental",
            "aeronautical", "production", "metallurgy", "plastic",
            "ceramic", "rubber", "food", "dairy", "marine"
        ]
        for cell in cells:
            if any(kw in cell.lower() for kw in branch_keywords):
                result["_program"] = cell.strip()
                result["branch"] = cell.strip()
                break

        if "branch" not in result:
            result["branch"] = current_program

        return result

    # If we have a long text cell, it might be an institute or program name
    for cell in cells:
        if len(cell) > 10:
            if any(kw in cell.lower() for kw in [
                "institute", "college", "university", "engg",
                "engineering", "tech"
            ]):
                return {"_institute": cell.strip(), "year": year, "_program": None}
            if any(kw in cell.lower() for kw in [
                "computer", "information", "civil", "mechanical", "electrical",
                "electronics", "chemical"
            ]):
                return {"_program": cell.strip(), "year": year, "_institute": None}

    return None


def parse_text_line(
    line: str, current_institute: str, current_program: str, year: str
) -> dict:
    """Parse a line of text (fallback when no tables found)."""
    # Similar logic to parse_table_row but for text lines
    parts = line.split()
    if len(parts) < 3:
        return None

    result = {"year": year, "_institute": None, "_program": None}

    # Check if line contains an institute name
    if any(kw in line.lower() for kw in [
        "institute", "college", "university", "engg"
    ]):
        result["_institute"] = line.strip()
        return result

    # Check for branch name
    branch_keywords = [
        "computer", "information technology", "civil", "mechanical",
        "electrical", "electronics", "chemical"
    ]
    if any(kw in line.lower() for kw in branch_keywords):
        result["_program"] = line.strip()
        return result

    # Check for rank data
    numbers = re.findall(r"\b\d{1,6}\b", line)
    if len(numbers) >= 2:
        result["opening_rank"] = int(numbers[-2])
        result["closing_rank"] = int(numbers[-1])
        result["college"] = current_institute
        result["branch"] = current_program

        # Try to find category
        for cat in ["General", "GEN", "OPEN", "OBC", "SEBC", "SC", "ST", "EWS", "TFWS"]:
            if cat.upper() in line.upper():
                result["category"] = normalize_category(cat)
                break
        else:
            result["category"] = "General"

        return result

    return None


def normalize_category(cat: str) -> str:
    """Normalize category names to standard form."""
    cat = cat.strip().upper()
    mapping = {
        "GENERAL": "General",
        "GEN": "General",
        "OPEN": "General",
        "1R": "General",
        "OBC": "OBC",
        "SEBC": "OBC",
        "2A": "OBC",
        "2B": "OBC",
        "SC": "SC",
        "3A": "SC",
        "ST": "ST",
        "3B": "ST",
        "EWS": "EWS",
        "TFWS": "TFWS",
        "PH": "PH",
        "NRI": "NRI",
        "AI": "AI",
    }
    return mapping.get(cat, None)


# ── Step 3: Combine and clean ─────────────────────────────────────────────────
def clean_and_save(all_dfs: list) -> pd.DataFrame:
    """Combine all year dataframes, clean, and save."""
    if not all_dfs:
        print("\n⚠️  No data extracted from PDFs!")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Clean college names
    df["college"] = df["college"].fillna("Unknown")
    df["college"] = df["college"].str.strip()

    # Clean branch names
    df["branch"] = df["branch"].fillna("Unknown")
    df["branch"] = df["branch"].str.strip()

    # Normalize branch names
    branch_map = {
        "computer engineering": "Computer Engineering",
        "computer science and engineering": "Computer Science",
        "computer science & engineering": "Computer Science",
        "computer science": "Computer Science",
        "information technology": "Information Technology",
        "electronics & communication engineering": "Electronics & Communication",
        "electronics and communication engineering": "Electronics & Communication",
        "electrical engineering": "Electrical Engineering",
        "mechanical engineering": "Mechanical Engineering",
        "civil engineering": "Civil Engineering",
        "chemical engineering": "Chemical Engineering",
    }

    for key, val in branch_map.items():
        mask = df["branch"].str.lower().str.contains(key, na=False)
        df.loc[mask, "branch"] = val

    # Remove rows with missing data
    df = df.dropna(subset=["closing_rank"])
    df = df[df["college"] != "Unknown"]
    df = df[df["branch"] != "Unknown"]

    # Ensure numeric types
    df["opening_rank"] = pd.to_numeric(df["opening_rank"], errors="coerce")
    df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce")

    # Sort
    df = df.sort_values(
        ["year", "college", "branch", "category"]
    ).reset_index(drop=True)

    # Save
    output_path = DATA_DIR / "acpc_cutoffs.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} records to {output_path}")
    print(f"\nDataset Summary:")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Colleges: {df['college'].nunique()}")
    print(f"  Branches: {df['branch'].nunique()}")
    print(f"  Categories: {sorted(df['category'].unique())}")
    print(f"\nSample data:")
    print(df.head(10).to_string(index=False))

    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🎓 ACPC Gujarat Engineering Cutoff Data Scraper")
    print("   Source: acpc.gujarat.gov.in")
    print("=" * 60)

    # Step 1: Download PDFs
    print("\n📥 Step 1: Downloading cutoff PDFs...")
    pdf_files = {}
    for year, url in TARGET_PDFS.items():
        filepath = download_pdf(url, year)
        if filepath:
            pdf_files[year] = filepath

    if not pdf_files:
        print("\n❌ No PDFs were downloaded. Check your internet connection.")
        sys.exit(1)

    # Step 2: Parse PDFs
    print("\n📊 Step 2: Parsing PDF tables...")
    all_dfs = []
    for year, filepath in sorted(pdf_files.items()):
        df = parse_cutoff_pdf(filepath, year)
        if not df.empty:
            all_dfs.append(df)

    # Step 3: Clean and save
    print("\n🧹 Step 3: Cleaning and saving data...")
    final_df = clean_and_save(all_dfs)

    if final_df.empty:
        print("\n⚠️  PDF parsing returned no structured data.")
        print("    This is common with ACPC PDFs as they use varied formats.")
        print("    Falling back to synthetic data generation based on ACPC trends...")
        print("    Run: python data/generate_data.py")
    else:
        print(f"\n🎉 Successfully scraped {len(final_df)} cutoff records!")
        print(f"   Output: data/acpc_cutoffs.csv")


if __name__ == "__main__":
    main()
