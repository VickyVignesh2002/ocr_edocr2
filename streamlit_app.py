import base64
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from pdf2image import convert_from_bytes

from edocr2 import tools


st.set_page_config(page_title="eDOCr2 OCR", layout="wide")


@st.cache_resource
def load_ocr_models(models_dir: str):
    import tensorflow as tf
    from edocr2.keras_ocr.detection import Detector
    from edocr2.keras_ocr.recognition import Recognizer

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    gdt_model = os.path.join(models_dir, "recognizer_gdts.keras")
    dim_model = os.path.join(models_dir, "recognizer_dimensions_2.keras")

    recognizer_gdt = Recognizer(alphabet=tools.ocr_pipelines.read_alphabet(gdt_model))
    recognizer_gdt.model.load_weights(gdt_model)

    alphabet_dim = tools.ocr_pipelines.read_alphabet(dim_model)
    recognizer_dim = Recognizer(alphabet=alphabet_dim)
    recognizer_dim.model.load_weights(dim_model)

    detector = Detector()

    return {
        "detector": detector,
        "recognizer_gdt": recognizer_gdt,
        "recognizer_dim": recognizer_dim,
        "alphabet_dim": alphabet_dim,
        "gpu_count": len(gpus),
    }


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


def build_comparison_image(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    target_h = max(original.shape[0], mask.shape[0])

    def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img
        new_w = int(img.shape[1] * (h / img.shape[0]))
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    left = resize_to_height(original, target_h)
    right = resize_to_height(mask, target_h)
    return cv2.hconcat([left, right])


def save_summary_txt(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as txt_file:
        txt_file.write(f"Input: {payload['file_path']}\n")
        txt_file.write(f"Output folder: {payload['output_dir']}\n")
        txt_file.write(f"Generated at: {payload['timestamp']}\n\n")
        txt_file.write(f"Table rows: {payload['table_rows']}\n")
        txt_file.write(f"GD&T items: {payload['gdt_count']}\n")
        txt_file.write(f"Dimension items: {payload['dimension_count']}\n")
        txt_file.write(f"Other info items: {payload['other_info_count']}\n")



def call_ollama_vision(model: str, image_bgr: np.ndarray, prompt: str) -> str:
    _, encoded = cv2.imencode(".png", image_bgr)
    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


st.title("eDOCr2 Streamlit OCR")
st.caption("Upload a drawing (PNG/JPG/PDF), run OCR, compare image vs mask, and download CSV/JSON/TXT outputs.")

models_dir = st.text_input("Models directory", value="edocr2/models")
language = st.text_input("Tesseract language", value="eng")
max_img_size = st.slider("Max image size (dimension OCR)", min_value=768, max_value=2048, value=1240, step=64)

use_ollama = st.checkbox("Enable Ollama post-processing", value=False)
ollama_model = st.text_input("Ollama model", value="granite3.2-vision:latest")
ollama_prompt = st.text_area(
    "Ollama prompt",
    value="Summarize key dimensions, tolerances, material references, and potential manufacturability notes from this engineering drawing.",
    height=100,
)

uploaded = st.file_uploader("Upload drawing", type=["png", "jpg", "jpeg", "pdf"])

if uploaded is not None and st.button("Run OCR", type="primary"):
    try:
        with st.spinner("Loading models..."):
            model_bundle = load_ocr_models(models_dir)

        gpu_msg = "GPU detected and enabled" if model_bundle["gpu_count"] > 0 else "GPU not detected, using CPU"
        st.info(gpu_msg)

        image = decode_upload(uploaded)
        original_image = image.copy()

        with st.spinner("Running segmentation..."):
            _, frame, gdt_boxes, tables, dim_boxes = tools.layer_segm.segment_img(
                image,
                autoframe=True,
                frame_thres=0.7,
                GDT_thres=0.02,
                binary_thres=127,
            )

        with st.spinner("Running OCR..."):
            process_img = image.copy()
            table_results, updated_tables, process_img = tools.ocr_pipelines.ocr_tables(tables, process_img, language)
            gdt_results, updated_gdt_boxes, process_img = tools.ocr_pipelines.ocr_gdt(
                process_img,
                gdt_boxes,
                model_bundle["recognizer_gdt"],
            )

            if frame:
                process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]

            dimensions, other_info, process_img, _ = tools.ocr_pipelines.ocr_dimensions(
                process_img,
                model_bundle["detector"],
                model_bundle["recognizer_dim"],
                model_bundle["alphabet_dim"],
                frame,
                dim_boxes,
                cluster_thres=20,
                max_img_size=max_img_size,
                language=language,
                backg_save=False,
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = Path(uploaded.name).stem
        output_dir = Path("results") / "streamlit" / f"{stem}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_img = tools.output_tools.mask_img(original_image, updated_gdt_boxes, updated_tables, dimensions, frame, other_info)

        table_results, gdt_results, dimensions, other_info = tools.output_tools.process_raw_output(
            str(output_dir),
            table_results,
            gdt_results,
            dimensions,
            other_info,
            save=True,
        )

        mask_path = output_dir / f"{stem}_mask.png"
        original_path = output_dir / f"{stem}_original.png"
        comparison_path = output_dir / f"{stem}_comparison.png"
        cv2.imwrite(str(mask_path), mask_img)
        cv2.imwrite(str(original_path), original_image)

        comparison = build_comparison_image(original_image, mask_img)
        cv2.imwrite(str(comparison_path), comparison)

        payload = {
            "file_path": uploaded.name,
            "output_dir": str(output_dir.resolve()),
            "timestamp": timestamp,
            "table_rows": sum(len(t) for t in table_results) if table_results else 0,
            "gdt_count": len(gdt_results) if gdt_results else 0,
            "dimension_count": len(dimensions) if dimensions else 0,
            "other_info_count": len(other_info) if other_info else 0,
            "table_results": table_results,
            "gdt_results": gdt_results,
            "dimension_results": dimensions,
            "other_info": other_info,
        }

        json_path = output_dir / "ocr_results.json"
        txt_path = output_dir / "ocr_results.txt"
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
        save_summary_txt(txt_path, payload)

        ollama_out_path = None
        if use_ollama:
            with st.spinner(f"Running Ollama model {ollama_model}..."):
                ollama_text = call_ollama_vision(ollama_model, comparison, ollama_prompt)
            ollama_out_path = output_dir / "ollama_summary.txt"
            ollama_out_path.write_text(ollama_text, encoding="utf-8")
            st.subheader("Ollama Summary")
            st.text_area("Local model output", value=ollama_text, height=200)

        st.success(f"OCR completed. Results saved to: {output_dir.resolve()}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("Mask")
            st.image(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("Comparison")
        st.image(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("Dimension CSV Preview")
        dim_csv_path = output_dir / "dimension_results.csv"
        if dim_csv_path.exists():
            with dim_csv_path.open("r", encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file)
                rows = list(reader)
            st.dataframe(rows)
        else:
            st.warning("dimension_results.csv was not generated.")

        st.subheader("Downloads")
        files_to_download = [
            output_dir / "table_results.csv",
            output_dir / "gdt_results.csv",
            output_dir / "dimension_results.csv",
            output_dir / "other_info.csv",
            json_path,
            txt_path,
            mask_path,
            comparison_path,
            original_path,
        ]
        if ollama_out_path:
            files_to_download.append(ollama_out_path)

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
