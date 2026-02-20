# eDOCr2 (Windows + Streamlit)

Engineering drawing OCR with segmentation, GD&T extraction, dimension OCR, and optional local Ollama post-processing.

## What is included

- CLI/sample runner: `test_drawing.py`
- Reusable OCR package: `edocr2/`
- Streamlit frontend: `streamlit_app.py`
- Persistent output saving to `results/`

## 1) Prerequisites (Windows)

- Python 3.11 recommended
- Git
- Tesseract OCR installed
- Poppler (`pdftoppm`) installed (for PDF input)
- Optional GPU for TensorFlow
- Optional Ollama (for local post-processing)

### Install system tools

```powershell
winget install --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements --silent
winget install --id oschwartz10612.Poppler --accept-package-agreements --accept-source-agreements --silent
```

Then restart terminal/VS Code once.

## 2) Clone and setup

### Option A: UV (recommended)

Windows:

```powershell
winget install --id=astral-sh.uv -e
git clone https://github.com/VickyVignesh2002/ocr_edocr2.git
cd ocr_edocr2
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Linux/Cloud:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/VickyVignesh2002/ocr_edocr2.git
cd ocr_edocr2
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option B: pip

```powershell
git clone https://github.com/VickyVignesh2002/ocr_edocr2.git
cd ocr_edocr2
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Using uv, replace package installs with:

```powershell
uv pip install -r requirements.txt
```

## 3) Download model files

Download from Releases and place in `edocr2/models/`:

- `recognizer_dimensions_2.keras`
- `recognizer_dimensions_2.txt`
- `recognizer_gdts.keras`
- `recognizer_gdts.txt`

Folder should look like:

```text
edocr2/models/
  recognizer_dimensions_2.keras
  recognizer_dimensions_2.txt
  recognizer_gdts.keras
  recognizer_gdts.txt
```

## 4) Run options

### Option A: Script runner

```powershell
python test_drawing.py
```

Input source is configured in `test_drawing.py`.

### Option B: Streamlit frontend (recommended)

```powershell
streamlit run streamlit_app.py
```

In the app:

1. Upload PNG/JPG/PDF
2. Enable **High accuracy mode** and **Preprocess image for OCR** for better results
3. Choose language and max image size
3. Click **Run OCR**
4. Download CSV/JSON/TXT/images from buttons

## 5) Output locations

### `test_drawing.py`

Outputs are saved to:

```text
results/<input_file_stem>/
```

Contains:

- `table_results.csv`
- `gdt_results.csv`
- `dimension_results.csv`
- `other_info.csv`
- `<stem>_mask.png`
- `ocr_results.json`
- `ocr_results.txt`

### `streamlit_app.py`

Outputs are saved to:

```text
results/streamlit/<input_stem>_<timestamp>/
```

Contains the same artifacts plus:

- `<stem>_original.png`
- `<stem>_comparison.png`
- `ollama_summary.txt` (when Ollama is enabled)
- `ollama_dimension_results.csv` (when Ollama refinement is enabled)

The app also prints and displays the exact output folder path every run.

## 6) GPU usage

- The OCR app checks TensorFlow GPUs and enables memory growth.
- If GPU is detected, Streamlit shows GPU status.
- If not, it falls back to CPU and still runs fully.

Important: on native Windows, TensorFlow GPU support may not be available depending on your install/runtime.
For highest chance of TensorFlow GPU acceleration, use WSL2 + CUDA configured Python environment.

Verify TensorFlow sees GPU:

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 7) Ollama local integration (no OpenAI required)

If Ollama is running locally, Streamlit can call vision model post-processing.

Start Ollama service and ensure a vision model exists, e.g.:

- `granite3.2-vision:latest`

In Streamlit:

- Enable **Ollama post-processing**
- Set model name
- Enable **Ollama dimension refinement CSV** for an extra pass focused on dimensional values
- Run OCR

## 8) Known notes

- TensorFlow deprecation messages are warnings, not failures.
- Large drawings can take long in dimension OCR on CPU.
- Swedish OCR (`swe`) requires Swedish Tesseract data installed; otherwise app may fallback based on available language packs.
