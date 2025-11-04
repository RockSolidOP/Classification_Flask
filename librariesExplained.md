# Libraries Explained (Plain English)

This project uses a few specialized libraries to turn PDF pages into searchable “fingerprints” and then suggest labels based on similar pages you’ve already curated.

## Core ML and Search

- open_clip_torch + torch
  - What it does: Turns a page image into a numeric fingerprint (a vector) that captures visual content.
  - Why it matters: These fingerprints let us compare pages by “looks like this,” not by file name or text alone.

- faiss-cpu
  - What it does: A very fast search engine for vectors. Given a new page’s fingerprint, it finds the most similar ones you’ve labeled.
  - Why it matters: Powers the “suggest labels” feature by retrieving close matches instantly.

## PDF Handling

- pypdfium2
  - What it does: Renders a PDF page into an image we can feed into the fingerprint step.

- PyMuPDF (fitz)
  - What it does: Reads PDF structure so we can access text, positions, and layout when needed during curation.

- pdfplumber
  - What it does: Helps extract text and tables from PDFs cleanly.

## Image Processing

- Pillow (PIL)
  - What it does: Basic image operations (open, convert, resize) used before creating fingerprints.

## How It Fits Together

1) Render a PDF page to an image (pypdfium2, Pillow)
2) Create the page’s fingerprint (open_clip_torch + torch)
3) Store/search fingerprints to find similar pages (faiss-cpu)
4) Use matches to suggest labels; PDF readers (PyMuPDF, pdfplumber) support curation tasks

Notes
- ML/search packages are listed in `requirements-ml.txt`.
- PDF and image utilities are in `requirements.txt`.
