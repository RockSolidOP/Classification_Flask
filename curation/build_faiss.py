from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    emb_parquet = root / "dataset" / "v2" / "embeddings" / "clip_vitb32.parquet"
    out_dir = root / "dataset" / "v2" / "faiss"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_index = out_dir / "clip_vitb32.index"
    out_idmap = out_dir / "id_map.jsonl"

    try:
        import faiss
    except Exception as e:
        raise SystemExit("faiss-cpu not installed. Add to requirements and install.") from e

    if not emb_parquet.exists():
        raise SystemExit(f"Embeddings file not found: {emb_parquet}")

    df = pd.read_parquet(emb_parquet)
    if df.empty:
        raise SystemExit("No embeddings to index.")

    vecs = np.stack(df["vector"].apply(lambda x: np.array(x, dtype="float32")).to_numpy())
    # Normalize for cosine similarity (inner product on normalized vectors)
    faiss.normalize_L2(vecs)

    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    faiss.write_index(index, str(out_index))

    with open(out_idmap, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            f.write(json.dumps({
                "offset": int(i),
                "id": row["id"],
                "label": row["label"],
                "base_label": row.get("base_label", ""),
                "page_in_form": int(row.get("page_in_form", 1)),
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {out_index} and {out_idmap}")


if __name__ == "__main__":
    main()

