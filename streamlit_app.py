import base64
import ast
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from pdf2image import convert_from_bytes


st.set_page_config(page_title="eDOCr2 OCR", layout="wide")


def decode_upload(uploaded_file) -> np.ndarray:
    if uploaded_file.name.lower().endswith(".pdf"):
        pages = convert_from_bytes(uploaded_file.getvalue(), first_page=1, last_page=1)
        image = np.array(pages[0])
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode uploaded file.")
    return image


def preprocess_for_ocr(image: np.ndarray, enable: bool, upscale: float) -> np.ndarray:
    if not enable:
        return image

    work = image.copy()
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if upscale > 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, sharpen_kernel)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def annotate_datapoints(image: np.ndarray, datapoints: list[dict]) -> np.ndarray:
    annotated = image.copy()
    for idx, item in enumerate(datapoints, start=1):
        x = item.get("x")
        y = item.get("y")
        if x is None or y is None:
            continue
        x_i, y_i = int(x), int(y)
        cv2.circle(annotated, (x_i, y_i), 6, (0, 255, 255), -1)
        label = item.get("label") or item.get("raw_text") or f"pt_{idx}"
        cv2.putText(
            annotated,
            str(label)[:40],
            (x_i + 8, max(16, y_i - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (50, 220, 50),
            1,
            cv2.LINE_AA,
        )
    return annotated


def save_summary_txt(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as txt_file:
        txt_file.write(f"Input: {payload['file_path']}\n")
        txt_file.write(f"Output folder: {payload['output_dir']}\n")
        txt_file.write(f"Generated at: {payload['timestamp']}\n\n")
        txt_file.write(f"Model: {payload['model']}\n")
        txt_file.write(f"Datapoints extracted: {payload['datapoints_count']}\n")



def _request_with_error_detail(url: str, payload: dict, timeout: int = 300) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code >= 400:
        detail = response.text
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                detail = parsed.get("error") or parsed.get("message") or json.dumps(parsed)
        except Exception:
            pass
        raise RuntimeError(f"Ollama request failed ({response.status_code}) at {url}: {detail}")
    return response.json()


def _resize_if_needed(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image_bgr
    scale = max_side / float(largest)
    new_w = int(width * scale)
    new_h = int(height * scale)
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def call_ollama_vision(model: str, image_bgr: np.ndarray, prompt: str, ollama_url: str, max_side: int) -> str:
    send_image = _resize_if_needed(image_bgr, max_side)

    def _call_once(img: np.ndarray) -> str:
        _, encoded_local = cv2.imencode(".png", img)
        image_b64_local = base64.b64encode(encoded_local.tobytes()).decode("utf-8")

        base_url = ollama_url.strip().rstrip("/")
        if base_url.endswith("/api/chat"):
            base_url = base_url[:-9]
        if base_url.endswith("/api/generate"):
            base_url = base_url[:-13]

        chat_url = f"{base_url}/api/chat"
        generate_url = f"{base_url}/api/generate"

        chat_payload = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64_local],
                }
            ],
        }
        generate_payload = {
            "model": model,
            "stream": False,
            "prompt": prompt,
            "images": [image_b64_local],
        }

        first_url, first_payload = (chat_url, chat_payload)
        second_url, second_payload = (generate_url, generate_payload)
        if ollama_url.strip().endswith("/api/generate"):
            first_url, first_payload = (generate_url, generate_payload)
            second_url, second_payload = (chat_url, chat_payload)

        first_error = None
        try:
            result = _request_with_error_detail(first_url, first_payload)
            if isinstance(result, dict):
                if isinstance(result.get("message"), dict):
                    content = result.get("message", {}).get("content", "")
                    if content:
                        return content
                response_text = result.get("response", "")
                if response_text:
                    return response_text
        except Exception as exc:
            first_error = str(exc)

        result = _request_with_error_detail(second_url, second_payload)
        if isinstance(result, dict):
            if isinstance(result.get("message"), dict):
                content = result.get("message", {}).get("content", "")
                if content:
                    return content
            response_text = result.get("response", "")
            if response_text:
                return response_text

        raise RuntimeError(
            "Ollama returned an unexpected response format. "
            f"First attempt error: {first_error}. "
            f"Second response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}"
        )

    try:
        return _call_once(send_image)
    except Exception as exc:
        retry_max_side = min(max_side, 1280)
        smaller = _resize_if_needed(send_image, retry_max_side)
        if smaller.shape[:2] != send_image.shape[:2]:
            return _call_once(smaller)
        raise RuntimeError(str(exc)) from exc


def extract_json_block(text: str) -> str:
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return text


def extract_balanced_json_candidate(text: str) -> str | None:
    start = -1
    opener = None
    for idx, ch in enumerate(text):
        if ch in "[{":
            start = idx
            opener = ch
            break
    if start == -1:
        return None

    closer = "]" if opener == "[" else "}"
    depth = 0
    in_string = False
    quote_char = ""
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote_char:
                in_string = False
            continue

        if ch in ['"', "'"]:
            in_string = True
            quote_char = ch
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def parse_ollama_json(raw_text: str):
    def _sanitize_candidate(text: str) -> str:
        sanitized = (
            text.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
            .replace("\r", "")
            .strip()
        )
        sanitized = re.sub(r'""\s*:', '":', sanitized)
        sanitized = re.sub(r':\s*""', ': "', sanitized)
        return sanitized

    base = extract_json_block(raw_text).strip()
    candidates = [base, raw_text.strip()]

    balanced_from_base = extract_balanced_json_candidate(base)
    balanced_from_raw = extract_balanced_json_candidate(raw_text)
    if balanced_from_base:
        candidates.append(balanced_from_base)
    if balanced_from_raw:
        candidates.append(balanced_from_raw)

    for candidate in candidates:
        if not candidate:
            continue

        normalized = _sanitize_candidate(candidate)

        if (normalized.startswith('"') and normalized.endswith('"')) or (
            normalized.startswith("'") and normalized.endswith("'")
        ):
            try:
                inner = ast.literal_eval(normalized)
                if isinstance(inner, str) and inner.strip() and inner != normalized:
                    candidates.append(_sanitize_candidate(inner))
            except Exception:
                pass

        try:
            return json.loads(normalized), "json"
        except Exception:
            pass

        no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", normalized)
        try:
            return json.loads(no_trailing_commas), "json_trailing_comma_fix"
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(no_trailing_commas)
            if isinstance(parsed, (dict, list)):
                return parsed, "python_literal"
        except Exception:
            pass

    return None, "failed"


def normalize_datapoints(parsed) -> list[dict]:
    def clean_key(key):
        return str(key).strip().strip('"').strip("'").strip()

    def clean_value(value):
        if isinstance(value, str):
            return value.strip().strip('"').strip("'").strip()
        return value

    if isinstance(parsed, dict):
        parsed = {clean_key(k): clean_value(v) for k, v in parsed.items()}
        for key in ["datapoints", "points", "items", "results", "data"]:
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]

    if not isinstance(parsed, list):
        return []

    normalized = []
    for item in parsed:
        if isinstance(item, str):
            normalized.append({
                "label": None,
                "value": None,
                "unit": None,
                "tolerance": None,
                "x": None,
                "y": None,
                "confidence": None,
                "raw_text": item,
            })
            continue

        if not isinstance(item, dict):
            continue

        item = {clean_key(k): clean_value(v) for k, v in item.items()}

        x = item.get("x")
        y = item.get("y")

        if isinstance(x, str) and re.fullmatch(r"-?\d+(\.\d+)?", x):
            x = float(x)
        if isinstance(y, str) and re.fullmatch(r"-?\d+(\.\d+)?", y):
            y = float(y)

        normalized.append({
            "label": item.get("label") or item.get("name") or item.get("field"),
            "value": item.get("value") or item.get("measurement") or item.get("nominal"),
            "unit": item.get("unit"),
            "tolerance": item.get("tolerance") or item.get("tol") or item.get("upper_tol") or item.get("lower_tol"),
            "x": int(x) if isinstance(x, (int, float)) else None,
            "y": int(y) if isinstance(y, (int, float)) else None,
            "confidence": item.get("confidence"),
            "raw_text": item.get("raw_text") or item.get("text"),
        })
    return normalized


st.title("eDOCr2 Streamlit OCR (Ollama GLM OCR)")
st.caption("Upload a drawing, extract structured datapoints with `glm-ocr:q8_0`, review table output, and download CSV/JSON/TXT.")

language = st.text_input("Language hint", value="eng")
high_accuracy = st.checkbox("High accuracy mode (slower)", value=True)
preprocess = st.checkbox("Preprocess image for OCR", value=True)
upscale_factor = st.select_slider("Upscale factor", options=[1.0, 1.25, 1.5, 2.0], value=1.5)
ollama_model = st.text_input("Ollama OCR model", value="glm-ocr:q8_0")
ollama_url = st.text_input("Ollama URL (base/chat/generate)", value="http://127.0.0.1:11434")
ollama_max_side = st.select_slider("Max image side sent to Ollama", options=[1024, 1280, 1536, 1792, 2048], value=1536)
ollama_prompt = st.text_area(
    "Extraction prompt",
    value=(
        "Extract engineering drawing datapoints and return ONLY JSON. "
        "Format: {\"datapoints\":[{\"label\":string|null,\"value\":string|null,\"unit\":string|null,"
        "\"tolerance\":string|null,\"x\":number|null,\"y\":number|null,\"confidence\":number|null,\"raw_text\":string|null}]}. "
        "Coordinates should be pixel positions in the image when possible. "
        f"Language hint: {language}."
    ),
    height=100,
)

uploaded = st.file_uploader("Upload drawing", type=["png", "jpg", "jpeg", "pdf"])

if uploaded is not None and st.button("Run OCR", type="primary"):
    try:
        image = decode_upload(uploaded)
        image = preprocess_for_ocr(image, preprocess, upscale_factor)
        original_image = image.copy()

        with st.spinner(f"Running {ollama_model} OCR extraction..."):
            raw_response = call_ollama_vision(ollama_model, image, ollama_prompt, ollama_url, ollama_max_side)
            parsed, parse_mode = parse_ollama_json(raw_response)
            if parsed is None:
                datapoints = normalize_datapoints([raw_response])
                st.warning(
                    "Model response was not strict JSON. "
                    "Saved raw output and added fallback row to table."
                )
            else:
                datapoints = normalize_datapoints(parsed)
                st.caption(f"Parsed model output mode: {parse_mode}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(uploaded.name).stem
        output_dir = Path("results") / "streamlit" / f"{stem}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        annotated_img = annotate_datapoints(original_image, datapoints)

        csv_path = output_dir / "datapoints.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["label", "value", "unit", "tolerance", "x", "y", "confidence", "raw_text"],
            )
            writer.writeheader()
            for row in datapoints:
                writer.writerow(row)

        raw_response_path = output_dir / "ollama_raw_response.txt"
        raw_response_path.write_text(raw_response, encoding="utf-8")

        original_path = output_dir / f"{stem}_original.png"
        annotated_path = output_dir / f"{stem}_annotated.png"
        cv2.imwrite(str(original_path), original_image)
        cv2.imwrite(str(annotated_path), annotated_img)

        payload = {
            "file_path": uploaded.name,
            "output_dir": str(output_dir.resolve()),
            "timestamp": timestamp,
            "model": ollama_model,
            "datapoints_count": len(datapoints),
            "datapoints": datapoints,
        }

        json_path = output_dir / "ocr_results.json"
        txt_path = output_dir / "ocr_results.txt"
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
        save_summary_txt(txt_path, payload)

        st.success(f"OCR completed. Results saved to: {output_dir.resolve()}")
        if not datapoints:
            st.warning("No structured datapoints were parsed. Check `ollama_raw_response.txt` and adjust prompt/model.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), width="stretch")
        with col2:
            st.subheader("Annotated datapoints")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), width="stretch")

        st.subheader("Datapoints table")
        st.dataframe(datapoints, width="stretch")

        st.subheader("Downloads")
        files_to_download = [
            csv_path,
            json_path,
            txt_path,
            original_path,
            annotated_path,
            raw_response_path,
        ]

        for file_path in files_to_download:
            if file_path.exists():
                with file_path.open("rb") as f:
                    st.download_button(
                        label=f"Download {file_path.name}",
                        data=f.read(),
                        file_name=file_path.name,
                        mime="application/octet-stream",
                    )

    except Exception as exc:
        st.exception(exc)
