#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_labels(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pages = data.get("pages", [])
    labels = {}
    for p in pages:
        page_no = int(p.get("page", 0))
        label = p.get("label", "Other")
        labels[page_no] = label
    total = int(data.get("total_pages", len(labels)))
    return total, labels


def write_csv(out_path: Path, total_pages: int, bookmark_labels: dict, manual_labels: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "bookmark_label", "manual_label"])
        for i in range(1, total_pages + 1):
            w.writerow([i, bookmark_labels.get(i, "Other"), manual_labels.get(i, "Other")])


def main():
    ap = argparse.ArgumentParser(description="Compare bookmark-derived labels to manual GT and write CSV")
    ap.add_argument("--bookmarks", required=True, help="Path to X_enhanced.json from CCH_ground_truth_bookmarks")
    ap.add_argument("--manual", required=True, help="Path to X_enhanced.json from CCH_ground_truth_corrected")
    ap.add_argument("--out", help="Output CSV path; defaults to CCH_Evaluation_Report/X_bookmarks_vs_gt.csv")
    args = ap.parse_args()

    bookmarks_path = Path(args.bookmarks)
    manual_path = Path(args.manual)
    if not bookmarks_path.exists() or not manual_path.exists():
        raise SystemExit("Input JSON path(s) do not exist")

    total_b, labels_b = load_labels(bookmarks_path)
    total_m, labels_m = load_labels(manual_path)
    total = max(total_b, total_m)

    if args.out:
        out_path = Path(args.out)
    else:
        report_dir = Path("CCH_Evaluation_Report")
        out_path = report_dir / f"{manual_path.stem}_bookmarks_vs_gt.csv"

    write_csv(out_path, total, labels_b, labels_m)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

