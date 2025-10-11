v1: Minimal corpus generator and scaffolding

What this does
- Builds a per-label corpus from ground truth in `CCH_ground_truth_corrected/` and PDFs in `CCH/`.
- Writes JSONL to `corpus/<family>/<label>.jsonl` (one line per page) with fields:
  - `doc`, `page`, `text`, `header_text`, `regex_flags`, `font_hist`.
- Creates `corpus_index.json` (counts/paths) and `label_profiles.json` (top terms per label).

Assumptions (v1)
- Require ground truth for any processed PDF (skip if missing).
- Extract native text via pdfminer.six; no OCR fallback in v1.
- Ground truth has been normalized once (e.g., GENRAL → GENERAL). No runtime mapping needed.
- Store normalized text (lowercase, collapsed whitespace).
- For now, process only the first GT+PDF pair (config `max_docs: 1`).

Quick start
1) Install dependencies (pdfminer.six; PyMuPDF optional fallback):
   pip install pdfminer.six pymupdf
2) Review config at `v1/config.json` (defaults should work for current repo).
3) Run the generator from repo root:
   python v1/generate_corpus.py --config v1/config.json

Outputs
- corpus/<family>/<label>.jsonl
- corpus_index.json
- label_profiles.json

Notes
- If the matching PDF `CCH/X.PDF` is missing or unreadable, the corresponding GT is skipped.
- Extend normalization and stopword lists as your dataset grows.
