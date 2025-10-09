#!/usr/bin/env python3
"""
Generate UltraTax ground-truth JSON from PDF bookmarks.

Defaults:
  - Input directory: `Source_PDF`
  - Output directory: `Ground_Truth`
  - Per-page family for bookmark-derived labels: `UltraTax`
  - Adds top-level `source: "UltraTax"` in the JSON

Usage examples:
  python extract_labels_ultratax.py                      # batch over `Source_PDF` -> `Ground_Truth`
  python extract_labels_ultratax.py --pdf Source_PDF/Foo.PDF  # single file
  python extract_labels_ultratax.py --overwrite          # allow overwriting existing outputs
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Try to import PDF reader libs; if unavailable, hardcode local venv paths.
def _try_import_pdfs() -> Tuple[object, object, object, object]:
    try:
        from PyPDF2 import PdfReader  # type: ignore
        from PyPDF2.generic import DictionaryObject, IndirectObject, ArrayObject  # type: ignore
        return PdfReader, DictionaryObject, IndirectObject, ArrayObject
    except ModuleNotFoundError:
        pass
    try:
        from pypdf import PdfReader  # type: ignore
        from pypdf.generic import DictionaryObject, IndirectObject, ArrayObject  # type: ignore
        return PdfReader, DictionaryObject, IndirectObject, ArrayObject
    except ModuleNotFoundError:
        pass

    # Augment sys.path with local venv site-packages
    repo_root = Path(__file__).resolve().parent
    candidates = []
    candidates.extend(repo_root.glob("venv/lib/python*/site-packages"))  # POSIX/macOS
    candidates.extend(repo_root.glob("venv/Lib/site-packages"))  # Windows
    for p in candidates:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    # Retry after path augmentation
    try:
        from PyPDF2 import PdfReader  # type: ignore
        from PyPDF2.generic import DictionaryObject, IndirectObject, ArrayObject  # type: ignore
        return PdfReader, DictionaryObject, IndirectObject, ArrayObject
    except ModuleNotFoundError:
        pass
    try:
        from pypdf import PdfReader  # type: ignore
        from pypdf.generic import DictionaryObject, IndirectObject, ArrayObject  # type: ignore
        return PdfReader, DictionaryObject, IndirectObject, ArrayObject
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing PyPDF2/pypdf. Either install it (e.g., in venv) or keep the repo's venv with site-packages present."
        ) from e


PdfReader, DictionaryObject, IndirectObject, ArrayObject = _try_import_pdfs()


def _resolve(obj):
    return obj.get_object() if isinstance(obj, IndirectObject) else obj


def _normalize_title_basic(title: str) -> str:
    # Treat commas as separators; remove them before collapsing to underscores
    title = title.replace(",", " ")
    norm = re.sub(r"[\s\-]+", "_", title.strip())
    norm = re.sub(r"_+", "_", norm).strip("_")
    return norm


def _roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in s.upper():
        v = vals.get(ch, 0)
        if v > prev:
            total += v - 2 * prev
        else:
            total += v
        prev = v
    return total


def _normalize_title(title: str) -> str:
    """Normalize bookmark titles using only the raw text.

    - Convert Page/Pg/P. N (or roman numerals) to P{N}.
    - Otherwise, sanitize the raw string: replace commas with spaces, collapse whitespace/dashes to underscores.
    - No heuristic mapping beyond what is explicitly present in the raw title.
    """
    raw = title.strip().replace("–", "-").replace("—", "-")
    low = raw.lower()

    # Page/Pg/P. + number or roman numeral -> P{n}
    m = re.match(r"^\s*(?:p(?:age)?|pg|p\.)\s*([ivxlcdm]+|\d+)\s*$", low)
    if m:
        tok = m.group(1)
        if tok.isdigit():
            return f"P{int(tok)}"
        try:
            return f"P{_roman_to_int(tok)}"
        except Exception:
            pass

    # Generic sanitization of the raw label
    return _normalize_title_basic(raw)


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


def _collect_outline_entries(first_node, page_ref_map, normalize: bool = True) -> List[Tuple[str, int, int]]:
    """Traverse the outline linked-list and collect (label, page_no, depth).

    If normalize is True, applies _normalize_title to the Title, otherwise keeps raw Title text.
    """
    entries: List[Tuple[str, int, int]] = []

    def node_to_entries(node_dict: DictionaryObject, depth: int):
        # Label
        label = None
        t = node_dict.get("/Title")
        if t is not None:
            try:
                raw_title = str(t)
                label = _normalize_title(raw_title) if normalize else raw_title.strip()
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


def build_bookmark_gt(pdf_path: Path, outdir: Path, family: str, config_name: str, source: str) -> Path:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    root = _resolve(reader.trailer.get("/Root"))
    outlines_indirect = root.get("/Outlines") if isinstance(root, DictionaryObject) else None

    # Default: Other/Other for every page
    pages = [
        {
            "page": i,
            "family": "Other",
            "label": "Other",          # current label (editable)
            "auto_label": "Other",      # auto-generated label (for restore)
            "updated_label": False,      # set True when user edits label
            "multipage": False,
            "raw_label": "Other",
        }
        for i in range(1, total_pages + 1)
    ]

    raw_pages = None
    if outlines_indirect:
        outlines = _resolve(outlines_indirect)
        first = outlines.get("/First") if outlines else None
        if first:
            page_ref_map = _build_page_ref_map(reader)
            # normalized entries for main pages array
            entries = _collect_outline_entries(first, page_ref_map, normalize=True)
            # raw entries for bookmarks view
            entries_raw = _collect_outline_entries(first, page_ref_map, normalize=False)
            # Choose deepest label per page
            best_for_page: Dict[int, Tuple[str, int]] = {}
            for label, page_no, depth in entries:
                cur = best_for_page.get(page_no)
                if cur is None or depth >= cur[1]:
                    best_for_page[page_no] = (label, depth)
            # Choose deepest raw label per page
            best_for_page_raw: Dict[int, Tuple[str, int]] = {}
            for label, page_no, depth in entries_raw:
                cur = best_for_page_raw.get(page_no)
                if cur is None or depth >= cur[1]:
                    best_for_page_raw[page_no] = (label, depth)
            for pno, (label, _depth) in best_for_page.items():
                if 1 <= pno <= total_pages:
                    raw_lbl = best_for_page_raw.get(pno, ("Other", 0))[0]
                    pages[pno - 1] = {
                        "page": pno,
                        "family": family,
                        "label": label,
                        "auto_label": label,
                        "updated_label": False,
                        "multipage": False,
                        "raw_label": raw_lbl,
                    }

            # Build raw pages array parallel to pages (original bookmark titles)
            raw_pages = [
                {"page": i, "family": "Other", "label": "Other"}
                for i in range(1, total_pages + 1)
            ]
            for pno, (label, _depth) in best_for_page_raw.items():
                if 1 <= pno <= total_pages:
                    raw_pages[pno - 1] = {"page": pno, "family": family, "label": label}

    # Post-process: Normalize sequential pages under a base section.
    # - When a base section (Form_* or Schedule_*) is seen, that page becomes Base_P1 (normalized).
    # - Subsequent P2/P3/... pages become Base_P2/Base_P3/... until a non-page label interrupts.
    # - Already-prefixed labels like Base_Pn keep their value and continue the prefix.
    def is_base_section(lbl: str) -> bool:
        # Treat any non-page, non-Other label as a base section
        if lbl == "Other":
            return False
        if re.match(r"^P\d+$", lbl):
            return False
        return True

    def is_page_suffix(lbl: str) -> bool:
        return bool(re.match(r"^P\d+$", lbl))

    def prefixed_page_match(lbl: str):
        # Generic: Any base label followed by _P{n}
        return re.match(r"^([A-Za-z0-9_()]+)_P(\d+)$", lbl)

    current_prefix = None
    for entry in pages:
        lbl = entry.get("label", "")
        fam = entry.get("family", "")
        if fam != family:
            continue

        m = prefixed_page_match(lbl)
        if m:
            current_prefix = m.group(1)
            continue

        if is_base_section(lbl):
            current_prefix = lbl
            new_val = f"{current_prefix}_P1"
            entry["label"] = new_val
            entry["auto_label"] = new_val
            continue

        if is_page_suffix(lbl) and current_prefix:
            new_val = f"{current_prefix}_{lbl}"
            entry["label"] = new_val
            entry["auto_label"] = new_val
            continue

        # interrupt on any other label
        current_prefix = None

    # Compute multipage flags: mark True for any Base_Pn group with size >= 2
    pages = _apply_multipage_flags(pages)

    data = {
        "document": pdf_path.name,
        "config": config_name,
        "source": source,
        "total_pages": total_pages,
        "pages": pages,
        "bookmarks": raw_pages
    }

    outdir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    base = pdf_path.stem
    date_tag = datetime.now().strftime("%b%d")  # e.g., Oct09
    out_path = outdir / f"{base}_{date_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def _apply_multipage_flags(pages: List[dict]) -> List[dict]:
    base_counts: Dict[str, int] = {}
    # First pass: count pages per base
    for e in pages:
        lbl = e.get("label", "")
        m = re.match(r"^(.+)_P(\d+)$", lbl)
        if m:
            base = m.group(1)
            base_counts[base] = base_counts.get(base, 0) + 1
    # Second pass: set flags
    for e in pages:
        lbl = e.get("label", "")
        m = re.match(r"^(.+)_P(\d+)$", lbl)
        if m and base_counts.get(m.group(1), 0) >= 2:
            e["multipage"] = True
        else:
            e["multipage"] = False
    return pages


def main():
    # Hardcoded, super-light defaults
    pdf_dir = Path("Source_PDF")
    outdir = Path("Ground_Truth")
    family = "UltraTax"
    config_name = "bookmarks_v2"
    source = "UltraTax"

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise SystemExit(f"Not a directory: {pdf_dir}")

    files: List[Path] = []
    files.extend(sorted(pdf_dir.glob("*.PDF")))
    files.extend(sorted(pdf_dir.glob("*.pdf")))

    if not files:
        print(f"No PDFs found in {pdf_dir}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    for pdf_path in files:
        out_path = build_bookmark_gt(pdf_path, outdir, family, config_name, source)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
