#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from calibration import fit_platt, save_platt


def _load_rows(index_path: Path) -> List[dict]:
    rows = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    # dedupe by id
    by_id = {}
    for r in rows:
        try:
            ident = f"{r.get('document')}#{int(r.get('page'))}"
        except Exception:
            continue
        by_id[ident] = r
    return [by_id[k] for k in sorted(by_id.keys())]


def _canonical(lbl: str, aliases: dict) -> str:
    return aliases.get(lbl, lbl)


def collect_scores(root: Path, rows: List[dict]) -> Tuple[List[float], List[int]]:
    # Import here to avoid heavy import cost when unused
    from suggest import _ensure_open_clip, _embed_image, search_neighbors  # type: ignore
    model, preprocess, torch = _ensure_open_clip()

    aliases = {}
    alias_path = root / "dataset" / "v1" / "aliases.json"
    try:
        if alias_path.exists():
            aliases = json.loads(alias_path.read_text(encoding="utf-8"))
    except Exception:
        aliases = {}

    scores: List[float] = []
    labels: List[int] = []
    for r in rows:
        img_path = r.get("image_path")
        lbl = _canonical(r.get("label", ""), aliases)
        if not img_path or not lbl:
            continue
        p = Path(img_path)
        if not p.exists():
            continue
        # Embed via OpenCLIP
        from PIL import Image
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        vec = _embed_image(img, model, preprocess, torch)
        import numpy as np
        # normalize for cosine similarity safety
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vec = vec.astype("float32")
        # Search neighbors (active FAISS)
        res = search_neighbors(root, vec, topk=6)
        # choose top-1 excluding self if present
        ident = f"{r.get('document')}#{int(r.get('page'))}"
        top = None
        for rr in res:
            if rr.get("id") == ident:
                continue
            top = rr
            break
        if not top and res:
            top = res[0]
        if not top:
            continue
        score = float(top.get("score", 0.0))
        pred_lbl = _canonical(top.get("label", ""), aliases)
        y = 1 if pred_lbl == lbl else 0
        scores.append(score)
        labels.append(y)
    return scores, labels


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--index", default=str(root / "dataset" / "v1" / "index" / "v1.jsonl"))
    ap.add_argument("--out", default=str(root / "dataset" / "v1" / "manifests" / "calibration.json"))
    ap.add_argument("--max", type=int, help="Optional cap on pages for speed")
    args = ap.parse_args()

    index_path = Path(args.index)
    rows = _load_rows(index_path)
    if args.max and args.max > 0:
        rows = rows[: args.max]
    scores, labels = collect_scores(root, rows)
    if not scores:
        raise SystemExit("No scores collected. Ensure embeddings/index exist and images are reachable.")
    model = fit_platt(scores, labels)
    save_platt(model, Path(args.out), note=f"active FAISS at build time")
    print(f"Saved calibration with A={model.A:.4f}, B={model.B:.4f}, n={model.n}, pos={model.pos}, neg={model.neg} -> {args.out}")


if __name__ == "__main__":
    main()

