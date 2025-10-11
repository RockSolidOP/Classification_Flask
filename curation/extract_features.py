from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any


def _try_import_fitz():
    try:
        import fitz  # PyMuPDF
        return fitz
    except Exception:
        return None


def _try_import_pdfium():
    try:
        import pypdfium2 as pdfium  # type: ignore
        return pdfium
    except Exception:
        return None


def _try_import_pdfplumber():
    try:
        import pdfplumber  # type: ignore
        return pdfplumber
    except Exception:
        return None


def extract_page_features(
    pdf_path: Path,
    page: int,
    images_root: Path,
    base_label: str,
    document_name: str,
    dpi: int = 300,
) -> Dict[str, Any]:
    """Extract image and word-level layout features.

    Prefers PyMuPDF (fitz) for both rendering and words+bboxes. Falls back to
    pypdfium2 (render) + pdfplumber (words) when fitz is unavailable. Finally,
    uses pdfplumber-only (no image) if needed.
    """
    fitz = _try_import_fitz()
    if fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
            p = doc.load_page(page - 1)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = p.get_pixmap(matrix=mat, alpha=False)

            # Save image under images_root/base_label/<doc>_<page>.png
            dest_dir = images_root / base_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            image_name = f"{document_name.replace('/', '_')}_{page}.png"
            image_path = dest_dir / image_name
            pix.save(str(image_path))

            width_px, height_px = pix.width, pix.height

            # Extract words in points
            try:
                words_info = p.get_text("words")
            except Exception:
                words_info = []

            words: List[str] = []
            boxes: List[List[int]] = []
            boxes_norm: List[List[int]] = []
            for w in words_info:
                if len(w) < 5:
                    continue
                x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
                if not text or str(text).isspace():
                    continue
                # points -> pixels using same zoom
                x0p = max(0, int(round(x0 * zoom)))
                y0p = max(0, int(round(y0 * zoom)))
                x1p = min(width_px, int(round(x1 * zoom)))
                y1p = min(height_px, int(round(y1 * zoom)))
                if x1p <= x0p or y1p <= y0p:
                    continue
                words.append(str(text))
                boxes.append([x0p, y0p, x1p, y1p])
                # normalize to 0..1000
                xn0 = int(round(x0p * 1000.0 / width_px))
                yn0 = int(round(y0p * 1000.0 / height_px))
                xn1 = int(round(x1p * 1000.0 / width_px))
                yn1 = int(round(y1p * 1000.0 / height_px))
                boxes_norm.append([xn0, yn0, xn1, yn1])

            page_size_pts = {"width": float(p.rect.width), "height": float(p.rect.height)}
            text_source = "pdf_text"
            text_len = sum(len(w) for w in words)

            doc.close()

            return {
                "image_path": str(image_path.as_posix()),
                "image_size": {"width_px": width_px, "height_px": height_px, "dpi": dpi},
                "words": words,
                "boxes": boxes,
                "boxes_norm": boxes_norm,
                "page_size_pts": page_size_pts,
                "text_source": text_source,
                "text_len": text_len,
            }
        except Exception:
            try:
                doc.close()
            except Exception:
                pass

    # Fallback: render with pypdfium2 and extract words with pdfplumber
    pdfium = _try_import_pdfium()
    pdfplumber = _try_import_pdfplumber()
    if pdfium is not None:
        try:
            doc = pdfium.PdfDocument(str(pdf_path))
            pg = doc[page - 1]
            scale = dpi / 72.0
            pil_image = pg.render(scale=scale).to_pil()

            dest_dir = images_root / base_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            image_name = f"{document_name.replace('/', '_')}_{page}.png"
            image_path = dest_dir / image_name
            pil_image.save(str(image_path))
            width_px, height_px = pil_image.size

            words: List[str] = []
            boxes: List[List[int]] = []
            boxes_norm: List[List[int]] = []
            page_size_pts = {"width": float(pg.get_size()[0]), "height": float(pg.get_size()[1])}
            text_source = "pdf_text"
            if pdfplumber is not None:
                try:
                    with pdfplumber.open(str(pdf_path)) as pdf:
                        p = pdf.pages[page - 1]
                        factor = dpi / 72.0
                        for w in p.extract_words(use_text_flow=True) or []:
                            txt = w.get("text", "")
                            if not txt or txt.isspace():
                                continue
                            x0 = int(round(float(w.get("x0", 0)) * factor))
                            y0 = int(round(float(w.get("top", 0)) * factor))
                            x1 = int(round(float(w.get("x1", 0)) * factor))
                            y1 = int(round(float(w.get("bottom", 0)) * factor))
                            x0 = max(0, min(x0, width_px))
                            y0 = max(0, min(y0, height_px))
                            x1 = max(0, min(x1, width_px))
                            y1 = max(0, min(y1, height_px))
                            if x1 <= x0 or y1 <= y0:
                                continue
                            words.append(str(txt))
                            boxes.append([x0, y0, x1, y1])
                            xn0 = int(round(x0 * 1000.0 / width_px))
                            yn0 = int(round(y0 * 1000.0 / height_px))
                            xn1 = int(round(x1 * 1000.0 / width_px))
                            yn1 = int(round(y1 * 1000.0 / height_px))
                            boxes_norm.append([xn0, yn0, xn1, yn1])
                except Exception:
                    pass

            text_len = sum(len(w) for w in words)
            return {
                "image_path": str(image_path.as_posix()),
                "image_size": {"width_px": width_px, "height_px": height_px, "dpi": dpi},
                "words": words,
                "boxes": boxes,
                "boxes_norm": boxes_norm,
                "page_size_pts": page_size_pts,
                "text_source": text_source,
                "text_len": text_len,
            }
        except Exception:
            pass

    # Last-resort: pdfplumber only (no image). Still extract words + boxes scaled by DPI.
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                p = pdf.pages[page - 1]
                width_pts, height_pts = float(p.width), float(p.height)
                width_px = int(round(width_pts * dpi / 72.0))
                height_px = int(round(height_pts * dpi / 72.0))
                words: List[str] = []
                boxes: List[List[int]] = []
                boxes_norm: List[List[int]] = []
                factor = dpi / 72.0
                for w in p.extract_words(use_text_flow=True) or []:
                    txt = w.get("text", "")
                    if not txt or txt.isspace():
                        continue
                    x0 = int(round(float(w.get("x0", 0)) * factor))
                    y0 = int(round(float(w.get("top", 0)) * factor))
                    x1 = int(round(float(w.get("x1", 0)) * factor))
                    y1 = int(round(float(w.get("bottom", 0)) * factor))
                    x0 = max(0, min(x0, width_px))
                    y0 = max(0, min(y0, height_px))
                    x1 = max(0, min(x1, width_px))
                    y1 = max(0, min(y1, height_px))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    words.append(str(txt))
                    boxes.append([x0, y0, x1, y1])
                    xn0 = int(round(x0 * 1000.0 / width_px))
                    yn0 = int(round(y0 * 1000.0 / height_px))
                    xn1 = int(round(x1 * 1000.0 / width_px))
                    yn1 = int(round(y1 * 1000.0 / height_px))
                    boxes_norm.append([xn0, yn0, xn1, yn1])
                text_len = sum(len(w) for w in words)
                return {
                    "image_size": {"width_px": width_px, "height_px": height_px, "dpi": dpi},
                    "words": words,
                    "boxes": boxes,
                    "boxes_norm": boxes_norm,
                    "page_size_pts": {"width": width_pts, "height": height_pts},
                    "text_source": "pdf_text",
                    "text_len": text_len,
                }
        except Exception:
            pass

    # Nothing available
    return {}
