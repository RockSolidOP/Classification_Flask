#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import requests


def _endpoint_and_key():
    # Load .env if available and map azure_key/azure_endpoint
    try:
        from dotenv import load_dotenv  # type: ignore
        from pathlib import Path as _P
        # Always load .env from repo root explicitly
        load_dotenv(dotenv_path=_P(__file__).resolve().parents[1] / ".env")
        if os.environ.get("azure_key") and not os.environ.get("AZURE_DOC_AI_KEY"):
            os.environ["AZURE_DOC_AI_KEY"] = os.environ["azure_key"]
        if os.environ.get("azure_endpoint") and not os.environ.get("AZURE_DOC_AI_ENDPOINT"):
            os.environ["AZURE_DOC_AI_ENDPOINT"] = os.environ["azure_endpoint"]
    except Exception:
        pass
    ep = os.environ.get("AZURE_DOC_AI_ENDPOINT")
    key = os.environ.get("AZURE_DOC_AI_KEY")
    if not ep or not key:
        raise SystemExit("Set AZURE_DOC_AI_ENDPOINT and AZURE_DOC_AI_KEY env vars")
    return ep.rstrip("/"), key


def _auth_headers(key: str, content_type: str) -> dict:
    return {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": content_type,
    }


def analyze_file(classifier_id: str, file_path: Path, content_type: str = "image/png") -> dict:
    endpoint, key = _endpoint_and_key()
    # Try classifier endpoint first
    url_cls = f"{endpoint}/formrecognizer/documentClassifiers/{classifier_id}:analyze?api-version=2023-07-31"
    headers = _auth_headers(key, content_type)
    with open(file_path, "rb") as f:
        data = f.read()
    resp = requests.post(url_cls, headers=headers, data=data, timeout=120)
    if resp.status_code == 404:
        # Some resources expose models instead of classifiers; try documentModels
        url_model = f"{endpoint}/formrecognizer/documentModels/{classifier_id}:analyze?api-version=2023-07-31"
        # For models, octet-stream is safest
        headers2 = _auth_headers(key, "application/octet-stream")
        resp2 = requests.post(url_model, headers=headers2, data=data, timeout=120)
        resp2.raise_for_status()
        return resp2.json()
    resp.raise_for_status()
    return resp.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier-id", help="If omitted, uses AZURE_DOC_AI_CLASSIFIER_ID or azure_classifier_id from .env")
    ap.add_argument("--file", required=True, help="Path to single-page PNG/JPG/PDF")
    ap.add_argument("--content-type", default="image/png")
    args = ap.parse_args()
    cid = args.classifier_id or os.environ.get("AZURE_DOC_AI_CLASSIFIER_ID") or os.environ.get("azure_classifier_id")
    if not cid:
        raise SystemExit("Provide --classifier-id or set AZURE_DOC_AI_CLASSIFIER_ID / azure_classifier_id in .env")
    result = analyze_file(cid, Path(args.file), args.content_type)
    print(result)


if __name__ == "__main__":
    main()
