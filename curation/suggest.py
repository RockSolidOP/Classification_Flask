from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np


_MODEL = None
_PREPROCESS = None
_TORCH = None
_INDEX = None
_IDMAP = []


def _ensure_open_clip():
    global _MODEL, _PREPROCESS, _TORCH
    if _MODEL is not None:
        return _MODEL, _PREPROCESS, _TORCH
    try:
        # Tame OpenMP/threading conflicts before heavy imports
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
        import open_clip
        import torch
    except Exception as e:
        raise RuntimeError("open_clip_torch/torch not installed")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.eval().to("cpu")
    _MODEL, _PREPROCESS, _TORCH = model, preprocess, torch
    return _MODEL, _PREPROCESS, _TORCH


def _ensure_faiss(root: Path):
    global _INDEX, _IDMAP
    if _INDEX is not None and _IDMAP:
        return _INDEX, _IDMAP
    try:
        import faiss
    except Exception:
        raise RuntimeError("faiss-cpu not installed")
    faiss_dir = root / "dataset" / "v1" / "faiss"
    active = None
    active_file = faiss_dir / "ACTIVE_VERSION.txt"
    if active_file.exists():
        try:
            active = active_file.read_text(encoding="utf-8").strip()
            if active and not active.startswith("_"):
                active = f"_{active}"
        except Exception:
            active = None
    index_path = faiss_dir / f"clip_vitb32{active or ''}.index"
    idmap_path = faiss_dir / f"id_map{active or ''}.jsonl"
    if not index_path.exists() or not idmap_path.exists():
        raise RuntimeError("FAISS index or id_map.jsonl not found; build Step 5 first")
    _INDEX = faiss.read_index(str(index_path))
    _IDMAP = []
    with open(idmap_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                _IDMAP.append(json.loads(line))
            except Exception:
                continue
    return _INDEX, _IDMAP


def _embed_image(img, model, preprocess, torch):
    with torch.no_grad():
        image = preprocess(img).unsqueeze(0)
        feats = model.encode_image(image)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy()


def embed_pdf_page(pdf_path: Path, page: int):
    from PIL import Image
    import pypdfium2 as pdfium
    model, preprocess, torch = _ensure_open_clip()
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        pg = doc[page - 1]
    except Exception:
        raise
    pil_image = pg.render().to_pil()
    vec = _embed_image(pil_image, model, preprocess, torch)
    # normalize for cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype("float32")


def search_neighbors(root: Path, query_vec: np.ndarray, topk: int = 5) -> List[Dict[str, Any]]:
    index, idmap = _ensure_faiss(root)
    import faiss
    D, I = index.search(query_vec.reshape(1, -1), topk)
    results: List[Dict[str, Any]] = []
    for rank in range(I.shape[1]):
        idx = int(I[0, rank])
        if idx < 0 or idx >= len(idmap):
            continue
        m = idmap[idx]
        results.append({
            "rank": rank + 1,
            "score": float(D[0, rank]),
            "id": m.get("id"),
            "label": m.get("label"),
            "base_label": m.get("base_label"),
            "page_in_form": m.get("page_in_form"),
        })
    return results
