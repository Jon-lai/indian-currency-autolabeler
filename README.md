# Indian Rupee Currency Banknote Auto-Labeler

An automated, high-precision banknote annotation and denomination detection pipeline powered by **Gemma-4-12B-it AWQ** (`Vishva007/gemma-4-12B-it-W4A16-AutoRound-AWQ`).

Optimized for **Google Colab GPU L4 (24GB VRAM)** and local RTX GPUs (12GB+ VRAM).

---

## Architecture: Two-Stage Cascaded Pipeline

The pipeline uses a two-stage cascade combining maximum bounding box precision with 100% recall:

```
                          Input Banknote Image
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │         STAGE 1: Class-Aware Text         │
             │       Targeted denomination prompt        │
             │         Single-Image Evaluation           │
             └─────────────────────┬─────────────────────┘
                                   │
                         Banknote Detected?
                                   │
                    ┌──────────────┴──────────────┐
                    │ YES                         │ NO (Missed / Complex)
                    ▼                             ▼
        ┌──────────────────────┐    ┌───────────────────────────┐
        │  Save Pascal VOC XML │    │  STAGE 2: Visual Fallback │
        │  & Preview Image     │    │  Target-First Multimodal  │
        │  (Status: stage1)    │    │  [Target, RefFront, Back] │
        └──────────────────────┘    └─────────────┬─────────────┘
                                                  │
                                        Banknote Detected?
                                                  │
                                   ┌──────────────┴──────────────┐
                                   │ YES                         │ NO
                                   ▼                             ▼
                       ┌──────────────────────┐      ┌──────────────────────┐
                       │  Save Pascal VOC XML │      │  Copy to missed/     │
                       │  & Preview Image     │      │  (Status: missed)    │
                       │  (Status: stage2)    │      └──────────────────────┘
                       └──────────────────────┘
```

1. **Stage 1 (Class-Aware Text Grounding)**:
   - Evaluates the target image using a targeted prompt describing that class's denomination, color palette, and cultural motifs.
   - Generates **tight, pixel-perfect bounding boxes** around the note at ~2.5s/image (handles 96%+ of all images).
2. **Stage 2 (Target-First Visual Reference Fallback)**:
   - Automatically triggered only for images missed in Stage 1.
   - Evaluates with the Target image placed **first** (`Index 0`) followed by official front & back reference images from `currency_references/`.
   - Recovers difficult edge cases (partial folds, extreme angles, obscure lighting) without coordinate drift.

---

## Supported Currency Classes

| Folder Name | Series | Denomination | Default Canonical Label | Cues & Motifs |
| :--- | :--- | :---: | :--- | :--- |
| `new10` | New MG Series | ₹10 | `10_rupee_note` | Chocolate Brown, Konark Sun Temple Wheel |
| `old10` | Older MG Series | ₹10 | `10_rupee_note` | Orange-Violet, Rhino / Elephant / Tiger |
| `new20` | New MG Series | ₹20 | `20_rupee_note` | Greenish-Yellow, Ellora Caves Kailash Temple |
| `old20` | Older MG Series | ₹20 | `20_rupee_note` | Reddish-Orange, Mount Harriet |
| `new50` | New MG Series | ₹50 | `50_rupee_note` | Fluorescent Cyan-Blue, Hampi with Chariot |
| `old50` | Older MG Series | ₹50 | `50_rupee_note` | Violet-Pink, Parliament of India |
| `old100` | Older MG Series | ₹100 | `100_rupee_note` | Blue-Green, Mount Kangchenjunga |
| `new100` | New MG Series | ₹100 | `100_rupee_note` | Lavender-Purple, Rani ki Vav |
| `new200` | New MG Series | ₹200 | `200_rupee_note` | Bright Orange-Yellow, Sanchi Stupa |
| `new500` | New MG Series | ₹500 | `500_rupee_note` | Stone Grey, Red Fort with Indian Flag |
| `new2000`| New MG Series | ₹2000 | `2000_rupee_note` | Magenta-Pink, Mangalyaan |

---

## Google Colab GPU L4 Setup Guide

### 1. Select the L4 Runtime in Colab
1. Open Google Colab.
2. In the top navigation bar, click **Runtime** > **Change runtime type**.
3. Under **Hardware accelerator**, select **GPU**.
4. Under **GPU type**, select **L4** (24GB VRAM).

### 2. Clone the Repository
```bash
!git clone https://github.com/Jon-lai/indian-currency-autolabeler.git
%cd indian-currency-autolabeler
```

### 3. Install Dependencies
```bash
!pip install -q -r requirements.txt
```

### 4. Provide Your Dataset
If your dataset is compressed on Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

!unzip -q /content/drive/MyDrive/indian_currency_640.zip -d /content/
```
Ensure your dataset contains class subfolders (e.g. `new10/`, `new20/`, etc.).

### 5. Run the Auto-Labeler

#### Quick Test (10 images per class):
```bash
!python gemma_autoround.py \
  --dataset-dir /content/indian_currency_640 \
  --mode cascade \
  --images-per-class 10 \
  --skip-existing
```

#### Full Run (All 5,400+ images):
```bash
!python gemma_autoround.py \
  --dataset-dir /content/indian_currency_640 \
  --mode cascade \
  --images-per-class 0 \
  --skip-existing
```

### 6. Download Annotations & Previews
```python
!zip -rq /content/dataset_annotations.zip labels previews annotation_report.csv
from google.colab import files
files.download('/content/dataset_annotations.zip')
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--dataset-dir` | str | `indian_currency_640` | Path to dataset containing class subfolders |
| `--ref-dir` | str | `currency_references` | Path to official banknote reference images |
| `--output-dir` | str | `labels` | Directory to save Pascal VOC XML annotations |
| `--preview-dir` | str | `previews` | Directory to save visual verification preview images |
| `--missed-dir` | str | `missed` | Directory to copy unannotated images |
| `--report-path` | str | `annotation_report.csv`| CSV summary of every processed image |
| `--mode` | str | `cascade` | `cascade` (Stage 1 text -> Stage 2 visual fallback), `text-only`, or `refs-only` |
| `--images-per-class`| int | `0` | Images to process per class (`0` for all images) |
| `--classes` | list | `None` | Specific classes to run (e.g. `--classes new10 old10`) |
| `--label-mode` | str | `denom` | Label style: `denom` (`10_rupee_note`) or `class` (`new10`) |
| `--skip-existing` | flag | `False` | Skip images that already have XML files |
| `--preview-count-per-class` | int | `15` | Previews to render per class |

---

## Repository Contents

```
├── gemma_autoround.py              # Main two-stage auto-labeling pipeline
├── colab_runner.ipynb              # One-click Jupyter notebook for Colab L4
├── download_currency_references.py # Downloader for official RBI reference images
├── currency_references/            # 24 front & back high-res official reference images
│   ├── inr_10_new_front.jpg
│   ├── inr_10_new_back.jpg
│   ├── banknotes_metadata.json
│   └── ...
├── requirements.txt                # Pip requirements for Colab & local Linux
├── pyproject.toml
└── README.md
```

---

## License
MIT License.
