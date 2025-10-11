from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", help="Embedding version tag to index (e.g., v3)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    emb_dir = root / "dataset" / "v2" / "embeddings"
    suffix = f"_{args.version}" if args.version else ""
    emb_parquet = emb_dir / f"clip_vitb32{suffix}.parquet"
    emb_jsonl = emb_dir / f"clip_vitb32{suffix}.jsonl"
    out_dir = root / "dataset" / "v2" / "faiss"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_index = out_dir / f"clip_vitb32{suffix}.index"
    out_idmap = out_dir / f"id_map{suffix}.jsonl"

    try:
        import faiss
    except Exception as e:
        raise SystemExit("faiss-cpu not installed. Add to requirements and install.") from e

    # Deduplicate by id, keep last occurrence
    by_id = {}

    if emb_parquet.exists():
        try:
            import pandas as pd  # type: ignore
            df = pd.read_parquet(emb_parquet)
            for _, row in df.iterrows():
            by_id[row["id"]] = {
                "label": row["label"],
                "base_label": row.get("base_label", ""),
                "page_in_form": int(row.get("page_in_form", 1)),
                "vector": np.array(row["vector"], dtype="float32"),
            }
        except Exception:
            # Fall back to JSONL if parquet cannot be read due to missing deps
            pass
    elif emb_jsonl.exists():
        with open(emb_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                by_id[rec.get("id")] = {
                    "label": rec.get("label", ""),
                    "base_label": rec.get("base_label", ""),
                    "page_in_form": int(rec.get("page_in_form", 1)),
                    "vector": np.array(rec.get("vector", []), dtype="float32"),
                }
    else:
        raise SystemExit(f"No embeddings found in {emb_dir}")

    if not by_id:
        raise SystemExit("No embeddings to index.")

    ids = list(by_id.keys())
    labels = [by_id[i]["label"] for i in ids]
    bases = [by_id[i]["base_label"] for i in ids]
    pnums = [by_id[i]["page_in_form"] for i in ids]
    vecs = np.stack([by_id[i]["vector"] for i in ids])
    faiss.normalize_L2(vecs)

    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    faiss.write_index(index, str(out_index))

    with open(out_idmap, "w", encoding="utf-8") as f:
        for i in range(len(ids)):
            f.write(json.dumps({
                "offset": int(i),
                "id": ids[i],
                "label": labels[i],
                "base_label": bases[i],
                "page_in_form": int(pnums[i]),
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {out_index} and {out_idmap}")


if __name__ == "__main__":
    main()
