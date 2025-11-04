# PDF Page Labeler + Curation (Flask)

A small Flask app to label PDF pages by content, edit/restore labels, and curate a training‑ready dataset (images + metadata) for downstream document classification. Optional similarity search (OpenCLIP + FAISS) can suggest labels based on nearest neighbors.

## Requirements

- Python 3.10–3.12 (tested locally on macOS)
- macOS/Linux/WSL2
- Optional for Suggestions: CPU‑only is fine. For embedding + FAISS features install extra ML deps (below).

## Environment Variables

Create a `.env` file in the project root (optional). The app will map `azure_*` to `AZURE_DOC_AI_*` if present.

```
# Optional Azure vars (not required for core features)
azure_key=your_azure_key
azure_endpoint=https://<your-azure-endpoint>.cognitiveservices.azure.com/
azure_classifier_id=optional_classifier_id

# Server bind (optional)
HOST=0.0.0.0
PORT=5000
```

Notes:
- Core upload/edit/curation features work fully offline.
- Keep your `.env` untracked; `python-dotenv` loads it automatically.

## Local Setup

1) Create and activate a virtualenv

```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2) Install base dependencies

```
pip install -r requirements.txt
```

3) (Optional) Install ML/search extras for Suggestions

```
pip install -r requirements-ml.txt
```

4) Verify environment loads

```
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('HOST', os.getenv('HOST','unset'))"
```

## Run the App

```
python app.py
```

- Open http://localhost:5000
- Upload a PDF on the Upload page; the app writes an enhanced mapping JSON to `Ground_Truth/` and lists it for quick access.
- Edit labels on the Edit page (change/restore). Multipage flags recompute on save.
- Curate single/all pages to build a dataset in `dataset/v1/` (JSONL index + images). See Curated view for counts and recent items.

## Features at a Glance

- Upload PDF → auto‑label pages; view and download the mapping JSON.
- Edit/restore labels; persistence to `Ground_Truth/*.json`.
- Curate pages → append rows to `dataset/v1/index/v1.jsonl` and save page images under `dataset/v1/images/<base_label>/`.
- Curated dashboard → totals and per‑label counts; manage/delete/dedupe records.
- Optional Suggestions → after building embeddings + FAISS, get similar pages for label hints.

## Dataset Versions

- Curated data is written under `dataset/v1/` by default. The `v1` folder name is the dataset version.
- The JSONL index lives at `dataset/v1/index/v1.jsonl`; images are stored under `dataset/v1/images/<base_label>/`.
- Embeddings and FAISS artifacts support their own version tags (independent of the dataset folder version):
  - Embeddings: `dataset/v1/embeddings/clip_vitb32[_vN].jsonl`
  - FAISS: `dataset/v1/faiss/clip_vitb32[_vN].index` and `dataset/v1/faiss/id_map[_vN].jsonl`
  - Active FAISS version: `dataset/v1/faiss/ACTIVE_VERSION.txt`

Changing dataset version
- The Flask app currently writes to `dataset/v1/` (see `app.py` → `DATASET_ROOT`). To start a new dataset version, create a new folder such as `dataset/v2/` and update `DATASET_ROOT` accordingly before curating new pages.

## Use in a Classifier Pipeline

Copy only what is needed for your current approach:

- Right now (k‑NN over FAISS; no model training):
  - Required artifacts to copy:
    - `dataset/v1/faiss/clip_vitb32[_vN].index`
    - `dataset/v1/faiss/id_map[_vN].jsonl`
    - `dataset/v1/faiss/ACTIVE_VERSION.txt`
  - Not required: `dataset/v1/images/`, `dataset/v1/index/v1.jsonl`, `dataset/v1/manifests/`, `dataset/v1/splits/`.
  - Runtime dependencies: `open_clip_torch`, `torch`, `pypdfium2`, `Pillow`, `faiss-cpu` (to render → embed → query FAISS).

- Future (after training a classifier such as LayoutLMv3):
  - For training a model: copy `dataset/vN/index/vN.jsonl` and `dataset/vN/images/` (splits/manifest recommended but optional).
  - For inference with the trained model: copy the model artifacts (weights + processor) produced by training; the FAISS index is not required unless you also want suggestions.

Path portability
- FAISS files are self‑contained. If you later train models and export JSONL, any `image_path` values are relative (e.g., `dataset/v1/images/...`); keep structure or rewrite paths as needed.

## Suggestions (Optional)

After installing `requirements-ml.txt`, build embeddings and an index:

```
# From repo root
python -m curation.build_embeddings            # writes dataset/v1/embeddings/clip_vitb32.jsonl
python -m curation.build_faiss                 # writes dataset/v1/faiss/*.index and id_map.jsonl

# Versioned runs (optional)
python -m curation.build_embeddings --version v2
python -m curation.build_faiss --version v2
```

Notes:
- The active FAISS version is controlled by `dataset/v1/faiss/ACTIVE_VERSION.txt`.
- If FAISS artifacts are missing, the Suggestions API returns HTTP 501.

## Docker (Optional)

The included `Dockerfile` can run the app in a container.

```
docker build -t page-labeler:latest .
docker run --rm -it \
  --env-file .env \
  -p 5000:5000 \
  -v "$(pwd)/Source_PDF:/app/Source_PDF" \
  -v "$(pwd)/Ground_Truth:/app/Ground_Truth" \
  -v "$(pwd)/dataset:/app/dataset" \
  page-labeler:latest
```

Open http://localhost:5000

## Project Layout

- `app.py` — Flask app (upload, edit, curate, curated dashboard, suggest API)
- `templates/` — HTML templates (`upload.html`, `edit.html`, `curated.html`, `help.html`)
- `curation/` — utilities for feature extraction, embeddings, FAISS, manifests/splits
  - `extract_features.py`, `build_embeddings.py`, `build_faiss.py`, `rerank.py`, `suggest.py`
- `Ground_Truth/` — enhanced mapping JSONs (output)
- `Source_PDF/` — uploaded PDFs (input)
- `dataset/v1/` — curated dataset root (`index/`, `images/`, `embeddings/`, `faiss/`, `manifests/`)
- `requirements.txt` — base deps; `requirements-ml.txt` — optional ML/search deps

## Troubleshooting

- Suggestions 501: Install `requirements-ml.txt` and build embeddings + FAISS.
- macOS threading/OpenMP warnings: mitigated via env flags set in `app.py`.
- PyMuPDF/PDF rendering issues: ensure `PyMuPDF`, `pypdfium2`, and `Pillow` are installed (from `requirements.txt`).
- Empty text on some PDFs: curation still saves images; OCR is optional and not required by the app.

## License

Proprietary code. Do not redistribute without permission.
