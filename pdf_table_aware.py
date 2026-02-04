from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import re
from pathlib import Path

import pdfplumber
import camelot

# ---------- helpers ----------
def chunk_text(text: str, max_chars=1800, overlap=200) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        i = max(0, j - overlap)
        if j == len(text):
            break
    return chunks

def likely_table_pages(pdf_path: str, max_pages=200) -> List[int]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages], start=1):
            text = page.extract_text() or ""
            digits = sum(ch.isdigit() for ch in text)
            if digits > 250:  # tune if needed
                pages.append(i)
    return pages

def is_good_table_df(df) -> bool:
    if df is None or df.empty:
        return False

    cells = df.astype(str).values.flatten().tolist()
    cells = [c.strip() for c in cells if c and c.strip() and c.strip().lower() != "nan"]

    if len(cells) < 12:
        return False

    avg_len = sum(len(c) for c in cells) / max(1, len(cells))
    if avg_len > 40:
        return False

    num_cells = sum(bool(re.search(r"\d", c)) for c in cells)
    numeric_ratio = num_cells / max(1, len(cells))
    if numeric_ratio < 0.15:
        return False

    nonempty_cols = 0
    for col in df.columns:
        col_vals = df[col].astype(str).str.strip()
        if (col_vals != "").sum() > 2:
            nonempty_cols += 1
    if nonempty_cols < 2:
        return False

    return True

# ---------- main API ----------
def extract_table_aware_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    pdf_path = str(pdf_path)
    out: List[Dict[str, Any]] = []

    # A) Narrative text chunks
    with pdfplumber.open(pdf_path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            for c in chunk_text(txt):
                out.append({
                    "type": "text",
                    "page": pageno,
                    "text": c,
                    "source": pdf_path,
                })

    # B) Table chunks (lattice first, stream fallback) + validation
    pages = likely_table_pages(pdf_path)
    for p in pages:
        tables = []
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(p), flavor="lattice")
        except Exception:
            tables = []

        if not tables:
            try:
                tables = camelot.read_pdf(
                    pdf_path, pages=str(p),
                    flavor="stream",
                    row_tol=12, edge_tol=500, strip_text="\n",
                )
            except Exception:
                continue

        for ti, t in enumerate(tables):
            df = t.df
            if not is_good_table_df(df):
                continue

            # keep whole table chunk OR row-group chunking
            # (here whole table for simplicity)
            out.append({
                "type": "table",
                "page": p,
                "table_id": f"{p}_{ti}",
                "text": df.to_csv(index=False),
                "source": pdf_path,
            })

    return out
