#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from pathlib import Path
import requests


def _endpoint_and_key():
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1]/'.env')
        if os.environ.get('azure_key') and not os.environ.get('AZURE_DOC_AI_KEY'):
            os.environ['AZURE_DOC_AI_KEY'] = os.environ['azure_key']
        if os.environ.get('azure_endpoint') and not os.environ.get('AZURE_DOC_AI_ENDPOINT'):
            os.environ['AZURE_DOC_AI_ENDPOINT'] = os.environ['azure_endpoint']
    except Exception:
        pass
    ep = os.getenv('AZURE_DOC_AI_ENDPOINT')
    key = os.getenv('AZURE_DOC_AI_KEY')
    if not ep or not key:
        raise SystemExit('Set AZURE_DOC_AI_ENDPOINT and AZURE_DOC_AI_KEY')
    return ep.rstrip('/'), key


def main():
    ep, key = _endpoint_and_key()
    h = { 'Ocp-Apim-Subscription-Key': key }
    urls = [
        ("classifiers", f"{ep}/formrecognizer/documentClassifiers?api-version=2023-07-31"),
        ("models", f"{ep}/formrecognizer/documentModels?api-version=2023-07-31"),
    ]
    out = {}
    for name, url in urls:
        try:
            r = requests.get(url, headers=h, timeout=30)
            r.raise_for_status()
            out[name] = r.json()
        except Exception as e:
            out[name] = {"error": str(e)}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

