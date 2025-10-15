#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import requests


def _endpoint_and_key():
    ep = os.environ.get("AZURE_DOC_AI_ENDPOINT")
    key = os.environ.get("AZURE_DOC_AI_KEY")
    if not ep or not key:
        raise SystemExit("Set AZURE_DOC_AI_ENDPOINT and AZURE_DOC_AI_KEY env vars")
    return ep.rstrip("/"), key


def analyze_file(classifier_id: str, file_path: Path, content_type: str = "image/png") -> dict:
    endpoint, key = _endpoint_and_key()
    url = f"{endpoint}/formrecognizer/documentClassifiers/{classifier_id}:analyze?api-version=2023-07-31"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": content_type,
    }
    with open(file_path, "rb") as f:
        resp = requests.post(url, headers=headers, data=f.read(), timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier-id", required=True)
    ap.add_argument("--file", required=True, help="Path to single-page PNG/JPG/PDF")
    ap.add_argument("--content-type", default="image/png")
    args = ap.parse_args()

    result = analyze_file(args.classifier_id, Path(args.file), args.content_type)
    print(result)


if __name__ == "__main__":
    main()

