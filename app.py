#!/usr/bin/env python3
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

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename

# Import the existing extraction logic
import extract_labels_ultratax as ultra

app = Flask(__name__)
app.secret_key = "dev-secret"

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "Source_PDF"
OUT_DIR = ROOT / "Ground_Truth"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


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

    # Generate the enhanced JSON using existing logic
    out_path = ultra.build_bookmark_gt(
        pdf_path=pdf_path,
        outdir=OUT_DIR,
        family="UltraTax",
        config_name="bookmarks_v2",
        source="UltraTax",
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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
