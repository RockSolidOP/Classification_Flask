# Azure Document Intelligence Classifier – Quick Guide

This doc explains how to train and use an Azure AI Document Intelligence (Document Intelligence / Form Recognizer) Custom Classifier for page‑level labels using the curated data produced by this repo.

## Overview

- We treat each page as a “document” (PNG/JPG or 1‑page PDF).
- Azure’s classifier is trained from label folders hosted in Azure Blob Storage (readable via SAS URL).
- After training, we call the analyze endpoint to get the predicted label and confidence for any page.

## Prerequisites

- Azure subscription + a Document Intelligence resource (you already have endpoint + key).
- Azure Storage account + a container for training data.
- Curated dataset with page images (created when you Curate in the app):
  - Images under `dataset/v2/images/<base_label>/<doc>_<page>.png`
  - Curated index at `dataset/v2/index/v2.jsonl`

## 1) Export label‑organized training data

Create a local tree with one folder per label, copying curated page images.

```
python azure/export_training_tree.py --out training_data --max-per-label 500
```

- Output: `training_data/<LABEL>/*.png`
- Tip: start with 200–500 images per label; exclude “Other”. Merge aliases first so labels are canonical.

## 2) Upload training data to Azure Blob and create a SAS URL

Using Azure CLI (or do the equivalent in Portal):

```
az login
# Create storage account and a container if needed
az storage account create -n <account> -g <resourceGroup> -l <region>
az storage container create --account-name <account> --name <container> --auth-mode login

# Upload your label folders
az storage blob upload-batch --account-name <account> -d <container> -s training_data --auth-mode login

# Generate a read SAS for the container (adjust expiry)
SAS=$(az storage container generate-sas --account-name <account> --name <container> --permissions rl --expiry 2030-01-01 --auth-mode login -o tsv)

# Container SAS base URL (use this in the build step)
echo https://<account>.blob.core.windows.net/<container>?$SAS
```

Notes
- Azure DI trains only from HTTPS‑accessible data (Blob + SAS is the standard path). Local folders/Azurite aren’t read by the cloud service.

## 3) Build the Azure Custom Classifier

Set env (do not commit your key):

```
export AZURE_DOC_AI_ENDPOINT=https://azuredocaiconversions.cognitiveservices.azure.com/
export AZURE_DOC_AI_KEY=YOUR_KEY
```

Kick off a build (labels auto‑discovered from your curated JSONL):

```
python azure/build_classifier.py \
  --classifier-id page-labels-v1 \
  --container-sas-base "https://<account>.blob.core.windows.net/<container>?$SAS"
```

What happens
- The script reads labels from `dataset/v2/index/v2.jsonl`.
- It builds a payload mapping each label to its folder SAS (`…/<label>?<SAS>`).
- It POSTs to `{endpoint}/formrecognizer/documentClassifiers:build?api-version=2023-07-31`.
- Response includes `Operation-Location`. Poll that URL (header `Ocp-Apim-Subscription-Key: $AZURE_DOC_AI_KEY`) until `status == succeeded`.

## 4) Classify a page (test locally)

Use a curated page image or a one‑page PDF:

```
python azure/analyze_page.py \
  --classifier-id page-labels-v1 \
  --file dataset/v2/images/<base_label>/<doc>_<page>.png \
  --content-type image/png
```

The JSON result includes the predicted label and confidence.

## How this integrates with the app

- We can add an “Azure Predict” button beside Suggest:
  - Render/locate the current page image.
  - Call the analyze API and display: `Azure predicts: <label> (0.xx)`.
  - Actions: Apply, Apply + Save, Apply + Curate, Merge Alias + Curate (same flow you already use).
- Batch scoring: a “Score with Azure” action in Curated can benchmark accuracy vs your labels.

## Tips for better models

- Balance classes: roughly equal examples per label; cap large classes (e.g., `--max-per-label 500`).
- Canonical labels: keep `dataset/v2/aliases.json` tidy and rebuild training data after merges.
- Page granularity: Decide if `_P1/_P2` are separate labels or train a base label model and track page index separately.
- Image quality: 150–300 DPI; ensure varied sources/years; avoid duplicates. Rebuild embeddings/index after major label changes.
- Iterate quickly: train a small model, review confusion, add more samples to hard classes.

## Troubleshooting

- “Cannot access training data”: Make sure the SAS URL is correct, not expired, and points to the container root; Azure appends label subfolders.
- “Empty classes”: Ensure each label folder has enough images (10+ as a bare minimum; 50–200 is better).
- “Low accuracy on near‑lookalikes”: Add more examples, normalize aliases, and/or split labels (e.g., `Form_1040` vs `Form_1040NR`).

## Files in this repo

- `azure/export_training_tree.py` — builds local label folders from curated index.
- `azure/build_classifier.py` — kicks off classifier build from a container SAS.
- `azure/analyze_page.py` — classifies a single file via the trained classifier.

