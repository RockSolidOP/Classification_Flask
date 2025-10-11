#!/usr/bin/env python3
import argparse
import re
import json
import sys
import os
import glob
from collections import defaultdict, Counter
from pathlib import Path

# Allow importing v1.utils when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import safe_name, normalize_family, normalize_text, tokenize, STOPWORDS, os_makedirs  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def iter_gt_files(gt_dir: Path):
    return sorted(gt_dir.glob("*_enhanced.json"))


def has_matching_pdf(gt_path: Path, pdf_dir: Path) -> bool:
    # GT document field may contain the exact PDF filename; else derive from basename
    try:
        data = json.loads(gt_path.read_text())
        doc = data.get("document")
        if doc:
            pdf_path = pdf_dir / doc
            if pdf_path.exists():
                return True
    except Exception:
        pass
    base = gt_path.name.replace("_enhanced.json", "")
    pdf_path = pdf_dir / f"{base}.PDF"
    return pdf_path.exists()


def resolve_pdf_path(gt_path: Path, pdf_dir: Path) -> Path:
    data = json.loads(gt_path.read_text())
    doc = data.get("document")
    if doc and (pdf_dir / doc).exists():
        return pdf_dir / doc
    base = gt_path.name.replace("_enhanced.json", "")
    p = pdf_dir / f"{base}.PDF"
    return p


def extract_page_text_pdfminer(pdf_path: Path, page_number: int) -> str:
    # page_number is 1-based
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer, LTTextBox, LTTextLine
    except Exception as e:
        raise RuntimeError("pdfminer.six is required for extraction in v1") from e

    texts = []
    for i, page_layout in enumerate(extract_pages(str(pdf_path))):
        if (i + 1) == page_number:
            for element in page_layout:
                if isinstance(element, (LTTextContainer, LTTextBox, LTTextLine)):
                    texts.append(element.get_text())
            break
    return "\n".join(texts)


def extract_page_features_pdfminer(pdf_path: Path, page_number: int, header_top_ratio: float):
    """Return dict with raw page text, header_text (top %), and font sizes.

    Uses pdfminer to parse a single page. header_top_ratio is fraction of page height
    considered the header region (e.g., 0.15 for top 15%).
    """
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTPage, LTTextContainer, LTTextBox, LTTextLine, LTChar
    except Exception as e:
        raise RuntimeError("pdfminer.six is required for extraction in v1") from e

    page_text_parts = []
    header_text_parts = []
    font_sizes = []

    for i, page_layout in enumerate(extract_pages(str(pdf_path))):
        if (i + 1) != page_number:
            continue
        # page bbox: (x0, y0, x1, y1)
        x0, y0, x1, y1 = page_layout.bbox
        height = max(1.0, (y1 - y0))
        header_y_threshold = y1 - header_top_ratio * height

        def walk(el):
            # Capture text and font sizes
            if isinstance(el, (LTTextContainer, LTTextBox, LTTextLine)):
                # Text
                try:
                    txt = el.get_text()
                except Exception:
                    txt = ""
                if txt:
                    page_text_parts.append(txt)
                    # header region check: use element bbox center y
                    try:
                        ex0, ey0, ex1, ey1 = el.bbox
                        cy = (ey0 + ey1) / 2.0
                        if cy >= header_y_threshold:
                            header_text_parts.append(txt)
                    except Exception:
                        pass
                # Traverse children to collect font sizes
                try:
                    for child in getattr(el, "_objs", []) or []:
                        walk(child)
                except Exception:
                    pass
                return
            if isinstance(el, LTChar):
                try:
                    if el.size:
                        font_sizes.append(float(el.size))
                except Exception:
                    pass
            # Recurse into generic containers that may have _objs
            for child in getattr(el, "_objs", []) or []:
                walk(child)

        # Walk page objects
        for element in getattr(page_layout, "_objs", []) or []:
            walk(element)
        break

    return {
        "text_raw": "\n".join(page_text_parts),
        "header_text_raw": "\n".join(header_text_parts),
        "font_sizes": font_sizes,
    }


def build_font_hist(font_sizes, bins):
    # bins are cut points; make labels for ranges like "<=8", "8-10", ">24"
    if not font_sizes:
        return {}
    edges = sorted(bins)
    labels = []
    labels.append(f"<= {edges[0]}")
    for a, b in zip(edges, edges[1:]):
        labels.append(f"{a}-{b}")
    labels.append(f"> {edges[-1]}")

    counts = [0] * len(labels)
    for s in font_sizes:
        if s <= edges[0]:
            counts[0] += 1
        elif s > edges[-1]:
            counts[-1] += 1
        else:
            for idx in range(len(edges) - 1):
                if edges[idx] < s <= edges[idx + 1]:
                    counts[idx + 1] += 1
                    break
    total = float(sum(counts)) or 1.0
    hist = {labels[i]: round(counts[i] / total, 6) for i in range(len(labels))}
    return hist


def load_label_registry(root: Path):
    path = root / "label.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def make_regex_flags(text_raw: str, registry: dict):
    flags = {}
    if not text_raw:
        return flags
    txt = text_raw
    # Known numeric form families from registry
    numeric_fams = set()
    schedule_codes = set()
    if registry:
        for fam in registry.get("families", []):
            if fam.isdigit():
                numeric_fams.add(fam)
        for fam, labels in registry.get("labels_by_family", {}).items():
            for lab in labels:
                # Look for tokens like *_Schedule_X* or *SCHEDULE_X*
                m = None
                if "Schedule_" in lab:
                    try:
                        part = lab.split("Schedule_")[1]
                        code = part.split("_")[0]
                        if code:
                            schedule_codes.add(code.upper())
                    except Exception:
                        pass
                elif "SCHEDULE_" in lab:
                    try:
                        part = lab.split("SCHEDULE_")[1]
                        code = part.split("_")[0]
                        if code:
                            schedule_codes.add(code.upper())
                    except Exception:
                        pass
    # Form flags
    for code in sorted(numeric_fams):
        pat = re.compile(rf"\bForm\s*{re.escape(code)}\b", re.IGNORECASE)
        flags[f"has_form_{code}"] = bool(pat.search(txt))
    # Schedule flags
    for code in sorted(schedule_codes):
        pat = re.compile(rf"\bSchedule\s*{re.escape(code)}\b", re.IGNORECASE)
        flags[f"has_schedule_{code.lower()}"] = bool(pat.search(txt))
    return flags


def write_jsonl(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Generate per-label corpus JSONL from GT + PDFs")
    ap.add_argument("--config", default="v1/config.json")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    root_dir = Path.cwd()
    pdf_dir = Path(cfg["source_pdfs"]).resolve()
    gt_dir = Path(cfg["ground_truth"]).resolve()
    out_dir = Path(cfg["output_dir"]).resolve()
    extractor = cfg.get("extractor", "pdfminer")
    require_gt = cfg.get("require_ground_truth", True)
    norm_map = cfg.get("normalization_map", {})
    text_ops = cfg.get("text_normalization", ["lowercase", "collapse_whitespace"])
    overwrite = cfg.get("overwrite_output", True)
    max_docs = cfg.get("max_docs")
    header_top_ratio = float(cfg.get("header_top_ratio", 0.15))
    font_bins = cfg.get("font_bins", [8, 10, 12, 14, 18, 24])

    if not pdf_dir.exists():
        sys.exit(f"Missing source_pdfs: {pdf_dir}")
    if not gt_dir.exists():
        sys.exit(f"Missing ground_truth: {gt_dir}")

    # Prepare outputs
    os_makedirs(str(out_dir))

    label_registry = load_label_registry(root_dir)
    index = {
        "documents": [],
        "totals": {"pages": 0, "docs": 0},
        "per_family": {},
        "per_label": {},
    }
    fam_counts = Counter()
    label_counts = Counter()
    label_term_counts = defaultdict(Counter)  # label -> term -> count

    processed_docs = 0
    for gt_path in iter_gt_files(gt_dir):
        if max_docs is not None and processed_docs >= max_docs:
            break
        if require_gt and not has_matching_pdf(gt_path, pdf_dir):
            continue

        data = json.loads(gt_path.read_text())
        pdf_path = resolve_pdf_path(gt_path, pdf_dir)
        if not pdf_path.exists():
            continue

        doc_entry = {"gt": str(gt_path.relative_to(gt_dir.parent)), "pdf": str(pdf_path.relative_to(gt_dir.parent))}
        pages_processed = 0

        for page in data.get("pages", []):
            page_num = int(page.get("page", 0))
            fam_raw = page.get("family", "Other")
            lab_raw = page.get("label", "Other")
            fam = normalize_family(fam_raw, norm_map)
            lab = lab_raw  # label typo normalization not specified yet

            try:
                if extractor == "pdfminer":
                    feats = extract_page_features_pdfminer(pdf_path, page_num, header_top_ratio)
                    text_raw = feats.get("text_raw", "")
                    header_raw = feats.get("header_text_raw", "")
                    font_sizes = feats.get("font_sizes", [])
                else:
                    # Default to pdfminer path
                    feats = extract_page_features_pdfminer(pdf_path, page_num, header_top_ratio)
                    text_raw = feats.get("text_raw", "")
                    header_raw = feats.get("header_text_raw", "")
                    font_sizes = feats.get("font_sizes", [])
            except Exception:
                continue

            text = normalize_text(text_raw, text_ops)
            # build enhanced record (formerly corpus_plus)
            record = {
                "doc": pdf_path.name,
                "page": page_num,
                "text": text,
                "header_text": normalize_text(header_raw, text_ops),
                "regex_flags": make_regex_flags(text_raw, label_registry),
                "font_hist": build_font_hist(font_sizes, font_bins),
            }

            fam_dir = out_dir / safe_name(fam)
            lab_file = fam_dir / f"{safe_name(lab)}.jsonl"
            os_makedirs(str(fam_dir))
            write_jsonl(lab_file, record)

            fam_counts[fam] += 1
            label_counts[f"{fam}/{lab}"] += 1
            pages_processed += 1

            # update label term counts for profiles
            for tok in tokenize(text):
                if tok in STOPWORDS or len(tok) <= 2:
                    continue
                label_term_counts[f"{fam}/{lab}"][tok] += 1

        if pages_processed > 0:
            processed_docs += 1
            index["documents"].append({
                "pdf": str(pdf_path),
                "gt": str(gt_path),
                "pages": pages_processed,
            })

    # finalize index
    index["totals"]["docs"] = processed_docs
    index["totals"]["pages"] = sum(fam_counts.values())
    index["per_family"] = dict(sorted(fam_counts.items()))
    index["per_label"] = dict(sorted(label_counts.items()))
    with open(out_dir / "corpus_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    # build label profiles (top terms)
    profiles = {}
    for lab_key, counter in label_term_counts.items():
        top = [(t, int(c)) for t, c in counter.most_common(50)]
        profiles[lab_key] = {"top_terms": top}
    with open(out_dir / "label_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(f"Processed docs: {processed_docs}")
    print(f"Wrote corpus under {out_dir}")


if __name__ == "__main__":
    main()
