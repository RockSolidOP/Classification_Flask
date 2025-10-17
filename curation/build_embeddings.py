from __future__ import annotations

import json
from pathlib import Path
from typing import List
import argparse


def _load_records(index_path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _load_aliases(root: Path) -> dict:
    p = root / "dataset" / "v1" / "aliases.json"
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _ensure_open_clip():
    try:
        import open_clip
        import torch
        return open_clip, torch
    except Exception as e:
        raise SystemExit("open_clip_torch/torch not installed. Add to requirements and install.") from e


def _load_model(device: str = "cpu"):
    open_clip, torch = _ensure_open_clip()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer, torch


def _embed_image(path: Path, model, preprocess, torch):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        image = preprocess(img).unsqueeze(0)
        feats = model.encode_image(image)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", help="Optional embedding version tag (e.g., v3)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    index_path = root / "dataset" / "v1" / "index" / "v1.jsonl"
    out_dir = root / "dataset" / "v1" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.version}" if args.version else ""
    out_parquet = out_dir / f"clip_vitb32{suffix}.parquet"
    out_jsonl = out_dir / f"clip_vitb32{suffix}.jsonl"

    rows = _load_records(index_path)
    aliases = _load_aliases(root)
    # Deduplicate by id, keep last occurrence
    by_id = {}
    for r in rows:
        try:
            ident = f"{r.get('document')}#{int(r.get('page'))}"
            by_id[ident] = r
        except Exception:
            continue
    if not rows:
        print("No curated records found.")
        return

    model, preprocess, _tokenizer, torch = _load_model()

    ids: List[str] = []
    labels: List[str] = []
    bases: List[str] = []
    pnums: List[int] = []
    vecs: List[list] = []

    for ident, r in by_id.items():
        img_path = r.get("image_path")
        if not img_path:
            continue
        p = Path(img_path)
        if not p.exists():
            continue
        vec = _embed_image(p, model, preprocess, torch)
        lbl = r.get("label", "")
        can = aliases.get(lbl, lbl)
        # if alias mapping changed, recompute base/page from canonical
        base = can
        pnum = 1
        if "_P" in can:
            try:
                base, pnum = can.rsplit("_P", 1)
                pnum = int(pnum)
            except Exception:
                base, pnum = can, 1
        ids.append(ident)
        labels.append(can)
        bases.append(base)
        pnums.append(int(pnum))
        vecs.append(vec.tolist())

    if not ids:
        print("No images embedded.")
        return
    
    # Always write JSONL (portable, no pyarrow dependency)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(len(ids)):
            rec = {
                "id": ids[i],
                "label": labels[i],
                "base_label": bases[i],
                "page_in_form": pnums[i],
                "vector": vecs[i],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out_jsonl} with {len(ids)} vectors")

    # Optionally write Parquet if pandas+pyarrow available
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame({
            "id": ids,
            "label": labels,
            "base_label": bases,
            "page_in_form": pnums,
            "vector": vecs,
        })
        df.to_parquet(out_parquet, index=False)
        print(f"Also wrote {out_parquet}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
