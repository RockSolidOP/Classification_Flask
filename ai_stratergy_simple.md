# AI Plan (Simple Version)

Use this as a quick guide to how we’ll label pages, save examples, and train a model. It’s written to be easy to scan and act on.

## What We’re Doing

- Classify pages: figure out what each page is (e.g., Form_1040_SS_P1).
- Save examples: keep a curated list of pages for each label.
- Train a model (LayoutLMv3) so it can recognize these pages later.

## What We Save For Each Page

- Where it is: document name + page number.
- The label: the name we want (e.g., Form_1040_SS_P2).
- The picture: a PNG image of the page.
- The words + positions: the text and where it sits on the page (needed for the model).
- Extras for tracking: who added it, when, and which JSON it came from.

## Where We Store It

- dataset/index: a running list (JSONL) of pages we added.
- dataset/images: the page images, grouped by label.
- dataset/faiss: a “similar pages” index so we can suggest labels.
- dataset/manifests: summary files (counts, versions).

## How New Pages Flow In

1) Auto-label in the app → you can edit labels.
2) Click “Add to Curated Dataset” → we save the page image, text/boxes, and a row in JSONL.
3) (Optional) Rebuild the “similar pages” index so suggestions improve.

## Why This Works

- Page is the unit we classify. Different PDFs can have different absolute page numbers; that’s fine.
- Labels are content-based (like Form_XXXX_P1). Pages with the same label look similar.
- The model (LayoutLMv3) uses the image + text + layout to learn what each label looks like.

## What You Can Do In The App

- Upload PDF → see labels.
- Edit labels, Save, Restore defaults.
- Preview a page (magnifying glass).
- (Next) Add selected pages to the curated dataset.
- (Next) See suggestions for a page based on similar pages.

## Checklist To Grow The Dataset

- [ ] Upload a PDF and review labels.
- [ ] Fix any wrong labels.
- [ ] Add pages to curated dataset.
- [ ] Rebuild “similar pages” (optional daily).
- [ ] Train/refresh the model when enough new pages are added.

## Tools Behind The Scenes (You Don’t Have To Run These Manually)

- PyMuPDF/pdfplumber: get text and positions.
- Tesseract (only if text is missing).
- FAISS: powers “similar pages”.
- LayoutLMv3: learns from image + text + layout.

## Terms (Plain English)

- JSONL: a text file where each line is one page’s info.
- Vector index (FAISS): a fast lookup to find pages that “look like this one”.
- Boxes: the rectangle for each word on the page (so the model knows the layout).

## What Success Looks Like

- You can quickly add pages to a label.
- The app suggests good labels for new pages.
- The model gets better as we add more curated pages.

## Step-by-Step Workflow Plan

1) Stabilize Auto-Labeling (Today)
- Confirm labels created by the app: `label`, `auto_label`, `updated_label`, `multipage`, `raw_label`.
- Ensure prefixing (Base_P1, Base_P2…) behaves as expected on your PDFs.
- Output JSON name format: `<file>_{MonDate}.json`.

2) Set Up Curated Dataset Skeleton
- Create folders: `dataset/v1/index`, `dataset/v1/images`, `dataset/v1/manifests`.
- Decide JSONL file name (e.g., `dataset/v1/index/v1.jsonl`).
- Define minimal per-page record: `document`, `page`, `label`, `auto_label`, `updated_label`, `base_label`, `page_in_form`, `multipage`, `raw_label`, `source_json`.

3) Add “Curate” Path From App
- In edit UI, add button “Add to Curated Dataset” per page and for all pages on screen.
- Backend endpoint: extracts features (next step) and appends a JSONL row + saves page image.

4) Feature Extraction For Each Curated Page
- Render page image (PyMuPDF) at ~300 DPI → save under `dataset/v1/images/<base_label>/<doc>_<page>.png`.
- Extract words + boxes (pdfplumber or PyMuPDF). If text is empty, OCR fallback (Tesseract).
- Normalize boxes to 0–1000 for LayoutLMv3.
- Append to JSONL: add `image_path`, `image_size`, `words`, `boxes`, `boxes_norm`, `page_size_pts`, `text_source`, `text_len`.

5) Build “Similar Pages” Index (Optional, High Impact)
- Compute image embeddings (CLIP ViT-B/32) for curated images; store as Parquet.
- Build FAISS index + id map under `dataset/v1/faiss/`.
- In the app, add a “Suggest labels” panel that queries the index for top-5 neighbors.

6) Create Dataset Manifest + Splits
- Generate `dataset/v1/manifests/v1.json` with counts per label, sources, tool versions, timestamp.
- Create doc-level stratified train/val/test split and save under `dataset/v1/splits/v1_splits.json`.

7) Train Baseline, Then LayoutLMv3
- Data readiness
  - Inputs: `dataset/v1/index/v1.jsonl` (deduped), `dataset/v1/splits/vN_splits.json` (doc-level splits)
  - Labels: start with fine‑grained `label` (e.g., `Form_1040_SR_P1`). Optionally switch to `base_label` later.
  - Filter for LayoutLMv3: keep only pages that have `image_path` + `words` + `boxes_norm`.

- Quick baseline (TF‑IDF + Linear)
  - Text features: header words + first N tokens from curated `words`; TF‑IDF (1–2‑grams), sublinear TF
  - Model: LinearSVC (or LogisticRegression)
  - Scripts: `training/train_tfidf.py` (train/eval/save), `training/predict_tfidf.py` (optional)
  - Artifacts: `models/baseline_vN/{model.joblib, vectorizer.joblib, label2id.json, metrics.json}`

- Primary (LayoutLMv3 fine‑tune)
  - Model: `microsoft/layoutlmv3-base` (HF Transformers)
  - Inputs: page `image_path`, `words`, `boxes_norm` (0–1000 ints), label → id
  - Scripts: `training/train_layoutlmv3.py` (processor+trainer, early stop, metrics)
  - Artifacts: `models/layoutlmv3_vN/` (model + processor + `label2id.json` + `metrics.json`)
  - Compute: prefer GPU (Colab/Azure). On CPU, run a tiny subset for sanity only.

- Metrics & tracking
  - Report macro‑F1, per‑label F1, confusion matrix on val/test
  - Save `run.json` with dataset version (vN), tool versions, and code commit for reproducibility

8) Inference Pipeline (For New PDFs)
- Given a PDF, render each page, extract words/boxes, run the trained model → predicted label.
- Show predictions in the app with confidence; allow quick corrections → feed back to curated set.

9) Versioning & Maintenance
- Append-only JSONL for new pages; bump dataset version when you collect a batch.
- Nightly (or weekly) job: rebuild FAISS index and update manifest.
- Keep reproducibility: record tool versions and commit hash in manifest.
