#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_rows(index_path: Path):
    rows = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    # Deduplicate by doc#page keep last
    by_id = {}
    for r in rows:
        try:
            ident = f"{r.get('document')}#{int(r.get('page'))}"
        except Exception:
            continue
        by_id[ident] = r
    return list(by_id.values())


def export_tree(index_path: Path, out_root: Path, max_per_label: int | None = None):
    rows = load_rows(index_path)
    out_root.mkdir(parents=True, exist_ok=True)
    per_label = {}
    copied = 0
    for r in rows:
        lbl = r.get("label")
        img = r.get("image_path")
        if not lbl or not img:
            continue
        if max_per_label is not None and per_label.get(lbl, 0) >= max_per_label:
            continue
        src = Path(img)
        if not src.exists():
            continue
        dst_dir = out_root / lbl
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        try:
            shutil.copy2(src, dst)
            per_label[lbl] = per_label.get(lbl, 0) + 1
            copied += 1
        except Exception:
            continue
    return copied, per_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-jsonl", default=str(Path(__file__).resolve().parents[1] / "dataset" / "v1" / "index" / "v1.jsonl"))
    ap.add_argument("--out", required=True, help="Output root that will contain one folder per label")
    ap.add_argument("--max-per-label", type=int, help="Optional cap per label (e.g., 500)")
    args = ap.parse_args()

    copied, per_label = export_tree(Path(args.index_jsonl), Path(args.out), args.max_per_label)
    print(f"Copied {copied} images")
    for lbl, n in sorted(per_label.items(), key=lambda kv: kv[0]):
        print(f"{lbl}: {n}")


if __name__ == "__main__":
    main()
