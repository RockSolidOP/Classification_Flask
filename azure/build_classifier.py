#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse, urlunparse
import requests
import json


def _endpoint_and_key():
    ep = os.environ.get("AZURE_DOC_AI_ENDPOINT")
    key = os.environ.get("AZURE_DOC_AI_KEY")
    if not ep or not key:
        raise SystemExit("Set AZURE_DOC_AI_ENDPOINT and AZURE_DOC_AI_KEY env vars")
    return ep.rstrip("/"), key


def _labels_from_jsonl(index_path: Path) -> List[str]:
    labels = []
    seen = set()
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            lbl = rec.get("label")
            if lbl and lbl not in seen and lbl != "Other":
                seen.add(lbl)
                labels.append(lbl)
    return labels


def _join_folder_sas(base_url: str, folder: str) -> str:
    # Insert folder path before SAS query if present
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/") + "/" + folder
    new = parsed._replace(path=path)
    return urlunparse(new)


def build_classifier(classifier_id: str, container_sas_base: str, labels: List[str]) -> dict:
    endpoint, key = _endpoint_and_key()
    url = f"{endpoint}/formrecognizer/documentClassifiers:build?api-version=2023-07-31"
    doc_types: Dict[str, dict] = {}
    for lbl in labels:
        doc_types[lbl] = {"azureBlobSource": {"containerUrl": _join_folder_sas(container_sas_base, lbl)}}
    payload = {
        "classifierId": classifier_id,
        "description": "Page classifier from curated images",
        "docTypes": doc_types,
    }
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    # Operation-Location header contains status URL; return both
    return {"status": resp.status_code, "operation": resp.headers.get("Operation-Location"), "body": resp.json() if resp.content else {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier-id", required=True)
    ap.add_argument("--container-sas-base", required=True, help="SAS URL to the container or folder root where label folders reside")
    ap.add_argument("--labels-file", help="Optional text file with one label per line")
    ap.add_argument("--index-jsonl", default=str(Path(__file__).resolve().parents[1] / "dataset" / "v2" / "index" / "v2.jsonl"))
    args = ap.parse_args()

    if args.labels_file:
        labels = [l.strip() for l in Path(args.labels_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        labels = _labels_from_jsonl(Path(args.index_jsonl))
    if not labels:
        raise SystemExit("No labels found. Provide --labels-file or ensure curated index has labels.")

    res = build_classifier(args.classifier_id, args.container_sas_base, labels)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

