#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def _augment_sys_path_for_venv():
    # Make local venv site-packages importable without activation
    here = Path(__file__).resolve().parent
    candidates = []
    candidates.extend(here.glob("venv/lib/python*/site-packages"))  # POSIX/macOS
    candidates.extend(here.glob("venv/Lib/site-packages"))  # Windows
    for p in candidates:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_augment_sys_path_for_venv()

# Tame OpenMP/threading issues on macOS when using Torch/FAISS/CLIP
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename

# Import the existing extraction logic
import extract_labels_ultratax as ultra

app = Flask(__name__)
app.secret_key = "dev-secret"

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "Source_PDF"
OUT_DIR = ROOT / "Ground_Truth"
DATASET_INDEX = ROOT / "dataset" / "v2" / "index" / "v2.jsonl"
DATASET_IMAGES = ROOT / "dataset" / "v2" / "images"
DATASET_ALIASES = ROOT / "dataset" / "v2" / "aliases.json"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_IMAGES.mkdir(parents=True, exist_ok=True)


def _load_aliases() -> dict:
    try:
        if DATASET_ALIASES.exists():
            import json
            with open(DATASET_ALIASES, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_aliases(aliases: dict) -> None:
    try:
        import json
        DATASET_ALIASES.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_ALIASES, "w", encoding="utf-8") as f:
            json.dump(aliases, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _canonicalize_label(label: str) -> str:
    aliases = _load_aliases()
    return aliases.get(label, label)


def _resolve_json_path(stem: str, name: str | None = None) -> Path | None:
    if name:
        p = OUT_DIR / name
        return p if p.exists() else None
    # else pick the most recent by mtime matching stem_*.json
    candidates = sorted(OUT_DIR.glob(f"{stem}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


@app.route("/")
def index():
    # List any existing enhanced JSON files for convenience
    existing = sorted(OUT_DIR.glob("*.json"))
    return render_template("upload.html", existing=[p.name for p in existing])


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("pdf")
    if not f or not f.filename.lower().endswith(".pdf"):
        flash("Please choose a PDF file.")
        return redirect(url_for("index"))
    filename = secure_filename(f.filename)
    pdf_path = SOURCE_DIR / filename
    f.save(str(pdf_path))
    # Read UltraTax mode checkbox (controls P1/P2 normalization)
    apply_prefix = bool(request.form.get("ultratax_mode"))

    # Generate the enhanced JSON using existing logic
    out_path = ultra.build_bookmark_gt(
        pdf_path=pdf_path,
        outdir=OUT_DIR,
        family="UltraTax",
        config_name="bookmarks_v2",
        source="UltraTax",
        apply_prefix=apply_prefix,
    )
    flash(f"Generated {out_path.name}")
    return redirect(url_for("edit", stem=pdf_path.stem, name=out_path.name))


@app.route("/edit/<stem>", methods=["GET", "POST"])
def edit(stem: str):
    name = request.args.get("name")
    # Allow passing filename directly via stem
    json_path = None
    if stem.endswith('.json'):
        json_path = OUT_DIR / stem
    else:
        json_path = _resolve_json_path(stem, name=name)
    if not json_path or not json_path.exists():
        flash("Mapping not found. Upload the PDF first.")
        return redirect(url_for("index"))

    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if request.method == "POST":
        # Update labels from form inputs named label_<page>
        changed = 0
        for entry in data.get("pages", []):
            pno = entry.get("page")
            field = f"label_{pno}"
            if field in request.form:
                new_label = request.form.get(field, "").strip()
                if new_label and new_label != entry.get("label"):
                    entry["label"] = new_label
                    entry["updated_label"] = True
                    changed += 1
        # Recompute multipage flags
        data["pages"] = ultra._apply_multipage_flags(data.get("pages", []))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        flash(f"Saved changes ({changed} labels updated)")
        return redirect(url_for("edit", stem=stem, name=json_path.name))

    return render_template(
        "edit.html",
        stem=stem,
        mapping=data,
        download_name=json_path.name,
    )


@app.route("/download/<path:filename>")
def download(filename: str):
    return send_from_directory(str(OUT_DIR), filename, as_attachment=True)

@app.route("/pdf/<path:filename>")
def serve_pdf(filename: str):
    # Serve source PDFs for previewing specific pages
    return send_from_directory(str(SOURCE_DIR), filename)


def _derive_base_and_page(label: str) -> tuple[str, int]:
    import re as _re
    m = _re.match(r"^(.+)_P(\d+)$", label)
    if m:
        return m.group(1), int(m.group(2))
    return label, 1


def _append_curated_record(mapping: dict, page_entry: dict, json_name: str) -> None:
    # Ensure dataset index dir exists
    DATASET_INDEX.parent.mkdir(parents=True, exist_ok=True)
    # Build minimal record
    # Canonicalize label via alias mapping
    cur_lbl = page_entry.get("label", "Other")
    can_lbl = _canonicalize_label(cur_lbl)
    base_label, page_in_form = _derive_base_and_page(can_lbl)
    record = {
        "document": mapping.get("document"),
        "page": int(page_entry.get("page")),
        "label": can_lbl,
        "auto_label": page_entry.get("auto_label", can_lbl),
        "updated_label": bool(page_entry.get("updated_label", False)),
        "base_label": base_label,
        "page_in_form": page_in_form,
        "multipage": bool(page_entry.get("multipage", False)),
        "raw_label": page_entry.get("raw_label", ""),
        "source_json": str((OUT_DIR / json_name).as_posix()),
    }
    # Remove any existing entries for this doc#page to avoid duplicates
    try:
        _ = _delete_curated_records(doc=record["document"], page=record["page"], delete_images=True)
    except Exception:
        pass
    # Optional: extract image + words/boxes (best effort)
    try:
        from curation.extract_features import extract_page_features
        pdf_path = SOURCE_DIR / mapping.get("document")
        feats = extract_page_features(
            pdf_path=pdf_path,
            page=int(page_entry.get("page")),
            images_root=DATASET_IMAGES,
            base_label=base_label,
            document_name=mapping.get("document"),
        )
        if feats:
            # Store image path relative to repo root for portability
            if "image_path" in feats:
                feats["image_path"] = str(Path(feats["image_path"]).as_posix())
            record.update(feats)
    except Exception:
        pass

    import json as _json
    with open(DATASET_INDEX, "a", encoding="utf-8") as f:
        f.write(_json.dumps(record, ensure_ascii=False))
        f.write("\n")


@app.post("/api/update_label/<stem>/<int:page>")
def api_update_label(stem: str, page: int):
    name = request.args.get("name")
    json_path = _resolve_json_path(stem, name=name)
    if not json_path or not json_path.exists():
        return jsonify({"ok": False, "error": "mapping not found"}), 404
    import json
    try:
        payload = request.get_json(force=True)
    except Exception:
        payload = {}
    new_label = (payload.get("label") or "").strip()
    if not new_label:
        return jsonify({"ok": False, "error": "label required"}), 400
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    updated = False
    for e in data.get("pages", []):
        if int(e.get("page")) == int(page):
            e["label"] = new_label
            e["updated_label"] = True
            updated = True
            break
    if not updated:
        return jsonify({"ok": False, "error": "page not found"}), 404
    # Recompute multipage flags and save
    data["pages"] = ultra._apply_multipage_flags(data.get("pages", []))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return jsonify({"ok": True})


@app.post("/api/restore_label/<stem>/<int:page>")
def api_restore_label(stem: str, page: int):
    name = request.args.get("name")
    json_path = _resolve_json_path(stem, name=name)
    if not json_path or not json_path.exists():
        return jsonify({"ok": False, "error": "mapping not found"}), 404
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    found = False
    restored = None
    for e in data.get("pages", []):
        if int(e.get("page")) == int(page):
            restored = e.get("auto_label", e.get("label"))
            e["label"] = restored
            e["updated_label"] = False
            found = True
            break
    if not found:
        return jsonify({"ok": False, "error": "page not found"}), 404
    data["pages"] = ultra._apply_multipage_flags(data.get("pages", []))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return jsonify({"ok": True, "label": restored})


@app.post("/api/restore_all/<stem>")
def api_restore_all(stem: str):
    name = request.args.get("name")
    json_path = _resolve_json_path(stem, name=name)
    if not json_path or not json_path.exists():
        return jsonify({"ok": False, "error": "mapping not found"}), 404
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for e in data.get("pages", []):
        e["label"] = e.get("auto_label", e.get("label"))
        e["updated_label"] = False
    data["pages"] = ultra._apply_multipage_flags(data.get("pages", []))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return jsonify({"ok": True})


@app.post("/api/curate/<stem>/<int:page>")
def api_curate_page(stem: str, page: int):
    """Append a single curated record for the given page to dataset/v2/index/v2.jsonl."""
    name = request.args.get("name")
    json_path = _resolve_json_path(stem, name=name)
    if not json_path or not json_path.exists():
        return jsonify({"ok": False, "error": "mapping not found"}), 404
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Find page entry
    entry = next((e for e in data.get("pages", []) if int(e.get("page")) == int(page)), None)
    if not entry:
        return jsonify({"ok": False, "error": "page not found"}), 404
    # Optional: skip obvious non-label pages
    if entry.get("label") == "Other":
        return jsonify({"ok": False, "error": "label is 'Other'"}), 400
    _append_curated_record(data, entry, json_path.name)
    return jsonify({"ok": True})


@app.post("/api/curate_all/<stem>")
def api_curate_all(stem: str):
    """Append curated records for all labeled pages in the mapping."""
    name = request.args.get("name")
    json_path = _resolve_json_path(stem, name=name)
    if not json_path or not json_path.exists():
        return jsonify({"ok": False, "error": "mapping not found"}), 404
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    count = 0
    for e in data.get("pages", []):
        if e.get("label") and e.get("label") != "Other":
            _append_curated_record(data, e, json_path.name)
            count += 1
    return jsonify({"ok": True, "count": count})


@app.post("/api/label_alias_merge")
def api_label_alias_merge():
    """Merge an alias label into a canonical label for future curation.

    Body: { alias: str, canonical: str }
    """
    payload = {}
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "invalid json"}), 400
    alias = (payload.get("alias") or "").strip()
    canonical = (payload.get("canonical") or "").strip()
    if not alias or not canonical:
        return jsonify({"ok": False, "error": "alias and canonical required"}), 400
    aliases = _load_aliases()
    aliases[alias] = canonical
    _save_aliases(aliases)
    return jsonify({"ok": True, "alias": alias, "canonical": canonical})


@app.route("/curated")
def curated():
    total = 0
    items = []
    # List embedding/index versions
    faiss_dir = ROOT / "dataset" / "v2" / "faiss"
    emb_dir = ROOT / "dataset" / "v2" / "embeddings"
    versions = []
    active_ver = None
    act_file = faiss_dir / "ACTIVE_VERSION.txt"
    if act_file.exists():
        try:
            active_ver = act_file.read_text(encoding="utf-8").strip()
        except Exception:
            active_ver = None
    # Detect versions by scanning embeddings jsonl
    import re
    for p in sorted(emb_dir.glob("clip_vitb32*.jsonl")):
        m = re.match(r"clip_vitb32(?:_(v\d+))?\.jsonl$", p.name)
        if m:
            ver = m.group(1) or "default"
            versions.append(ver)
    if DATASET_INDEX.exists():
        from collections import deque
        import json
        dq = deque(maxlen=200)
        with open(DATASET_INDEX, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                dq.append(line)
        for line in dq:
            try:
                rec = json.loads(line)
                items.append(rec)
            except Exception:
                continue
    return render_template("curated.html", total=total, items=items, versions=versions, active_ver=active_ver)


def _delete_curated_records(doc: str, page: int | None = None, delete_images: bool = True) -> dict:
    """Filter out curated records for a given doc (and page if provided).

    Returns stats: {deleted, kept}. Also deletes image files when available and delete_images is True.
    """
    import json
    import os
    from tempfile import NamedTemporaryFile

    if not DATASET_INDEX.exists():
        return {"deleted": 0, "kept": 0}

    deleted = 0
    kept = 0
    with open(DATASET_INDEX, "r", encoding="utf-8") as src, NamedTemporaryFile("w", delete=False, dir=str(DATASET_INDEX.parent), encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        for line in src:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                rec = json.loads(line_s)
            except Exception:
                # keep unparsable lines
                tmp.write(line)
                kept += 1
                continue
            if rec.get("document") == doc and (page is None or int(rec.get("page", -1)) == int(page)):
                deleted += 1
                # Best-effort delete image
                if delete_images:
                    img = rec.get("image_path")
                    if img:
                        try:
                            Path(img).unlink(missing_ok=True)
                        except Exception:
                            pass
                continue
            # keep
            tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    # Atomic replace
    os.replace(tmp_path, DATASET_INDEX)
    return {"deleted": deleted, "kept": kept}


@app.post("/api/curated_delete")
def api_curated_delete():
    """Delete a curated record by doc and optional page.

    Body (JSON) or query args: {doc: string, page?: int, delete_images?: bool}
    """
    payload = {}
    try:
        payload = request.get_json(force=False, silent=True) or {}
    except Exception:
        payload = {}
    doc = (payload.get("doc") or request.args.get("doc") or "").strip()
    page = payload.get("page") or request.args.get("page")
    delete_images = payload.get("delete_images")
    if delete_images is None:
        delete_images = request.args.get("delete_images", "1") in ("1", "true", "True")
    if not doc:
        return jsonify({"ok": False, "error": "doc required"}), 400
    page_int = None
    if page is not None and str(page).strip() != "":
        try:
            page_int = int(page)
        except Exception:
            return jsonify({"ok": False, "error": "invalid page"}), 400
    stats = _delete_curated_records(doc=doc, page=page_int, delete_images=bool(delete_images))
    return jsonify({"ok": True, **stats})


@app.post("/api/build_embeddings")
def api_build_embeddings():
    # Compute next version vN based on existing files
    emb_dir = ROOT / "dataset" / "v2" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    import re, subprocess
    max_n = 0
    for p in emb_dir.glob("clip_vitb32_*.jsonl"):
        m = re.match(r"clip_vitb32_v(\d+)\.jsonl$", p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    ver = f"v{max_n+1}"
    cmd = [sys.executable, str((ROOT / "curation" / "build_embeddings.py")), "--version", ver]
    try:
        subprocess.run(cmd, check=True)
        return jsonify({"ok": True, "version": ver})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/build_index")
def api_build_index():
    ver = request.form.get("version") or request.args.get("version")
    if not ver:
        return jsonify({"ok": False, "error": "version required"}), 400
    import subprocess
    cmd = [sys.executable, str((ROOT / "curation" / "build_faiss.py")), "--version", ver]
    try:
        subprocess.run(cmd, check=True)
        return jsonify({"ok": True, "version": ver})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/set_active_version")
def api_set_active_version():
    ver = request.form.get("version") or request.args.get("version")
    if not ver:
        return jsonify({"ok": False, "error": "version required"}), 400
    faiss_dir = ROOT / "dataset" / "v2" / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    (faiss_dir / "ACTIVE_VERSION.txt").write_text(ver, encoding="utf-8")
    return jsonify({"ok": True, "version": ver})


def _delete_all_curated(delete_images: bool = True) -> dict:
    """Delete all curated records (and optionally all curated images)."""
    total = 0
    if DATASET_INDEX.exists():
        # Count lines then truncate file
        with open(DATASET_INDEX, "r", encoding="utf-8") as f:
            for _ in f:
                total += 1
        # Truncate
        open(DATASET_INDEX, "w", encoding="utf-8").close()
    images_deleted = 0
    if delete_images and DATASET_IMAGES.exists():
        for p in DATASET_IMAGES.rglob("*"):
            try:
                if p.is_file():
                    p.unlink()
                    images_deleted += 1
            except Exception:
                pass
    return {"deleted": total, "images_deleted": images_deleted}


@app.post("/api/curated_delete_all")
def api_curated_delete_all():
    delete_images = request.args.get("delete_images", "1") in ("1", "true", "True")
    stats = _delete_all_curated(delete_images=delete_images)
    return jsonify({"ok": True, **stats})


@app.post("/api/curated_dedupe")
def api_curated_dedupe():
    """Deduplicate curated index by keeping the last entry per document#page."""
    import json
    from tempfile import NamedTemporaryFile
    if not DATASET_INDEX.exists():
        return jsonify({"ok": True, "deleted": 0, "kept": 0})
    # First pass: find last index per id
    positions = {}
    lines = []
    with open(DATASET_INDEX, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            lines.append(line)
            try:
                rec = json.loads(line)
                key = f"{rec.get('document')}#{int(rec.get('page', -1))}"
                positions[key] = idx
            except Exception:
                pass
    # Second pass: write only lines whose idx equals last position
    kept = 0
    with NamedTemporaryFile("w", delete=False, dir=str(DATASET_INDEX.parent), encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        for idx, line in enumerate(lines):
            try:
                rec = json.loads(line)
                key = f"{rec.get('document')}#{int(rec.get('page', -1))}"
                if positions.get(key) == idx:
                    tmp.write(line)
                    kept += 1
            except Exception:
                # keep unparsable lines
                tmp.write(line)
                kept += 1
    deleted = len(lines) - kept
    os.replace(tmp_path, DATASET_INDEX)
    return jsonify({"ok": True, "deleted": deleted, "kept": kept})


@app.route("/help")
def help_page():
    return render_template("help.html")


@app.get("/api/suggest/<stem>/<int:page>")
def api_suggest(stem: str, page: int):
    """Return top-k similar curated pages for the given PDF page.

    Requires FAISS index built (Step 5). If unavailable, returns 501.
    """
    try:
        from curation.suggest import embed_pdf_page, search_neighbors
    except Exception:
        return jsonify({"ok": False, "error": "suggestion engine not available"}), 501

    name = request.args.get("name")
    json_path = _resolve_json_path(stem, name=name)
    if not json_path or not json_path.exists():
        return jsonify({"ok": False, "error": "mapping not found"}), 404
    import json as _json
    with open(json_path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    pdf_path = SOURCE_DIR / data.get("document")
    if not pdf_path.exists():
        return jsonify({"ok": False, "error": "source PDF not found"}), 404
    try:
        q = embed_pdf_page(pdf_path, page)
        results = search_neighbors(ROOT, q, topk=10)
        # Apply alias canonicalization to suggestion labels and dedupe by canonical
        aliases = _load_aliases()
        agg = {}
        for r in results:
            lbl = r.get("label", "")
            can = aliases.get(lbl, lbl)
            # keep best score per canonical label
            if can not in agg or float(r.get("score", 0.0)) > float(agg[can].get("score", 0.0)):
                nr = dict(r)
                nr["label"] = can
                agg[can] = nr
        # Sort by score desc and cap to top 5
        merged = sorted(agg.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)[:5]
        # Re-rank
        for i, m in enumerate(merged, start=1):
            m["rank"] = i
        return jsonify({"ok": True, "results": merged})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Bind to configurable host/port to avoid conflicts (e.g., AirPlay using 5000)
    import os
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=True)
