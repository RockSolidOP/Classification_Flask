from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


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
    root = Path(__file__).resolve().parents[1]
    index_path = root / "dataset" / "v2" / "index" / "v2.jsonl"
    out_dir = root / "dataset" / "v2" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "clip_vitb32.parquet"

    rows = _load_records(index_path)
    if not rows:
        print("No curated records found.")
        return

    model, preprocess, _tokenizer, torch = _load_model()

    ids: List[str] = []
    labels: List[str] = []
    bases: List[str] = []
    pnums: List[int] = []
    vecs: List[list] = []

    for r in rows:
        img_path = r.get("image_path")
        if not img_path:
            continue
        p = Path(img_path)
        if not p.exists():
            continue
        vec = _embed_image(p, model, preprocess, torch)
        ids.append(f"{r.get('document')}#{int(r.get('page'))}")
        labels.append(r.get("label", ""))
        bases.append(r.get("base_label", ""))
        pnums.append(int(r.get("page_in_form", 1)))
        vecs.append(vec.tolist())

    if not ids:
        print("No images embedded.")
        return

    df = pd.DataFrame({
        "id": ids,
        "label": labels,
        "base_label": bases,
        "page_in_form": pnums,
        "vector": vecs,
    })
    df.to_parquet(out_parquet, index=False)
    print(f"Wrote {out_parquet} with {len(df)} vectors")


if __name__ == "__main__":
    main()

