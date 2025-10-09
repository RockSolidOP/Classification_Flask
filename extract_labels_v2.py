#!/usr/bin/env python3
"""
Generate a bookmark-derived ground-truth JSON (enhanced schema) for a PDF.

Usage:
  python extract_labels_v2.py --pdf CCH/SomeDoc.PDF \
      [--outdir CCH_ground_truth_bookmarks] [--family BOOKMARKS] [--config bookmarks_v2]

Output:
  Writes CCH_ground_truth_bookmarks/SomeDoc_enhanced.json with fields:
    - document: PDF filename
    - config: config name (default: bookmarks_v2)
    - total_pages: number of pages in the PDF
    - pages: array of {page, family, label}; default Other/Other unless a bookmark label exists
      for the page, in which case family=<family> (default BOOKMARKS) and label=<normalized label>

Notes:
  - For pages with multiple bookmark labels, the deepest (most specific) label wins.
  - Labels are normalized (spaces/hyphens -> single underscore, collapse repeats, trim).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from PyPDF2 import PdfReader
from PyPDF2.generic import DictionaryObject, IndirectObject, ArrayObject


def _resolve(obj):
    return obj.get_object() if isinstance(obj, IndirectObject) else obj


def _normalize_title(title: str) -> str:
    norm = re.sub(r"[\s\-]+", "_", title.strip())
    norm = re.sub(r"_+", "_", norm).strip("_")
    return norm


def _page_number_from_dest(dest, page_ref_map: Dict[Tuple[int, int], int]):
    """Return 1-based page number from a /Dest array's first element, if resolvable."""
    try:
        dest = _resolve(dest)
        if isinstance(dest, ArrayObject) and dest:
            first = dest[0]
            if isinstance(first, IndirectObject):
                key = (first.idnum, first.generation)
                if key in page_ref_map:
                    return page_ref_map[key] + 1
    except Exception:
        pass
    return None


def _build_page_ref_map(reader: PdfReader) -> Dict[Tuple[int, int], int]:
    page_ref_map: Dict[Tuple[int, int], int] = {}
    try:
        for idx, page in enumerate(reader.pages):
            ref = getattr(page, "indirect_ref", None) or getattr(page, "indirect_reference", None)
            if isinstance(ref, IndirectObject):
                page_ref_map[(ref.idnum, ref.generation)] = idx
    except Exception:
        pass
    return page_ref_map


def _collect_outline_entries(first_node, page_ref_map) -> List[Tuple[str, int, int]]:
    """Traverse the outline linked-list and collect (label, page_no, depth)."""
    entries: List[Tuple[str, int, int]] = []

    def node_to_entries(node_dict: DictionaryObject, depth: int):
        # Label
        label = None
        t = node_dict.get("/Title")
        if t is not None:
            try:
                label = _normalize_title(str(t))
            except Exception:
                label = None

        # Page
        page_no = None
        dest = node_dict.get("/Dest")
        if dest is None:
            action = node_dict.get("/A")
            action = _resolve(action) if action is not None else None
            if isinstance(action, DictionaryObject):
                dest = action.get("/D")
        if dest is not None:
            page_no = _page_number_from_dest(dest, page_ref_map)

        if label and page_no is not None:
            entries.append((label, int(page_no), depth))

        # Children
        first_child = node_dict.get("/First")
        if first_child is not None:
            cur = _resolve(first_child)
            while isinstance(cur, DictionaryObject):
                node_to_entries(cur, depth + 1)
                next_sib = cur.get("/Next")
                if next_sib is None:
                    break
                cur = _resolve(next_sib)

    # Iterate siblings starting from first_node
    cur = _resolve(first_node)
    while isinstance(cur, DictionaryObject):
        node_to_entries(cur, depth=0)
        next_sib = cur.get("/Next")
        if next_sib is None:
            break
        cur = _resolve(next_sib)

    return entries


def build_bookmark_gt(pdf_path: Path, outdir: Path, family: str, config_name: str) -> Path:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    root = _resolve(reader.trailer.get("/Root"))
    outlines_indirect = root.get("/Outlines") if isinstance(root, DictionaryObject) else None

    # Default: Other/Other for every page
    pages = [{"page": i, "family": "Other", "label": "Other"} for i in range(1, total_pages + 1)]

    if outlines_indirect:
        outlines = _resolve(outlines_indirect)
        first = outlines.get("/First") if outlines else None
        if first:
            page_ref_map = _build_page_ref_map(reader)
            entries = _collect_outline_entries(first, page_ref_map)
            # Choose deepest label per page
            best_for_page: Dict[int, Tuple[str, int]] = {}
            for label, page_no, depth in entries:
                cur = best_for_page.get(page_no)
                if cur is None or depth >= cur[1]:
                    best_for_page[page_no] = (label, depth)
            for pno, (label, _depth) in best_for_page.items():
                if 1 <= pno <= total_pages:
                    pages[pno - 1] = {"page": pno, "family": family, "label": label}

    data = {
        "document": pdf_path.name,
        "config": config_name,
        "total_pages": total_pages,
        "pages": pages,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    base = pdf_path.stem
    out_path = outdir / f"{base}_enhanced.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Extract bookmark-based ground truth JSON")
    ap.add_argument("--pdf", help="Path to a single input PDF (optional)")
    ap.add_argument("--pdf_dir", default="CCH", help="Directory containing PDFs to process (default: CCH)")
    ap.add_argument("--outdir", default="CCH_ground_truth_bookmarks", help="Output directory for JSON files (default: CCH_ground_truth_bookmarks)")
    ap.add_argument("--family", default="BOOKMARKS", help="Family to assign to bookmark-derived labels")
    ap.add_argument("--config", dest="config_name", default="bookmarks_v2", help="Config name to embed in JSON")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present")
    ap.add_argument("--pattern", default="*.PDF", help="Glob pattern for PDFs when using --pdf_dir (case-sensitive)")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            raise SystemExit(f"PDF not found: {pdf_path}")
        out_path = outdir / f"{pdf_path.stem}_enhanced.json"
        if out_path.exists() and not args.overwrite:
            print(f"Skip (exists): {out_path}")
        else:
            out_path = build_bookmark_gt(pdf_path, outdir, args.family, args.config_name)
            print(f"Wrote {out_path}")
        return

    # Directory mode
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise SystemExit(f"Not a directory: {pdf_dir}")

    # gather files with both .PDF and .pdf by default
    patterns = [args.pattern]
    if args.pattern == "*.PDF":
        patterns.append("*.pdf")

    files = []
    for pat in patterns:
        files.extend(sorted(pdf_dir.glob(pat)))

    if not files:
        print(f"No PDFs found in {pdf_dir} matching {patterns}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    failed = 0
    for pdf_path in files:
        try:
            out_path = outdir / f"{pdf_path.stem}_enhanced.json"
            if out_path.exists() and not args.overwrite:
                print(f"Skip (exists): {out_path}")
                skipped += 1
                continue
            out_path = build_bookmark_gt(pdf_path, outdir, args.family, args.config_name)
            print(f"Wrote {out_path}")
            written += 1
        except Exception as e:
            print(f"Failed {pdf_path}: {e}")
            failed += 1

    print(f"Done. Written: {written}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
