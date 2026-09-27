# ==============================================================================
# CLASS-AWARE CASCADED BANKNOTE AUTO-LABELER
# Model: Vishva007/gemma-4-12B-it-W4A16-AutoRound-AWQ
# Optimized for Google Colab GPU L4 (24GB) & RTX 3060 (12GB)
# ==============================================================================
# Two-Stage Cascaded Pipeline:
#   Stage 1: Fast Class-Aware Text Prompting (Pixel-perfect bounding boxes for 96%+)
#   Stage 2: Automatic Visual Reference Fallback for any missed / difficult images
#            (Target-First Multimodal Grounding with Front & Back Reference Images)
# ==============================================================================
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import csv
import glob
import json
import shutil
import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

# ============================ DEFAULT CONFIG ============================
MODEL_ID = "Vishva007/gemma-4-12B-it-W4A16-AutoRound-AWQ"

DEFAULT_DATASET_DIR = "indian_currency_640"
DEFAULT_REF_DIR     = "currency_references"
DEFAULT_OUTPUT_DIR  = "labels"
DEFAULT_PREVIEW_DIR = "previews"
DEFAULT_MISSED_DIR  = "missed"
DEFAULT_ERROR_DIR   = "errors"
DEFAULT_REPORT_PATH = "annotation_report.csv"
DEFAULT_RAW_LOG     = "raw_model_outputs.jsonl"

MAX_NEW_TOKENS   = 128
APPLY_NMS        = True
NMS_IOU          = 0.5

# Enable TensorFloat-32 (TF32) on Ampere / Ada Lovelace (RTX 3060 / L4 / A100)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================ CURRENCY REGISTRY ============================
CURRENCY_CLASSES = {
    "new10": {
        "name": "₹10 rupee banknote",
        "series": "New Mahatma Gandhi Series",
        "denom": "10",
        "canonical_label": "10_rupee_note",
        "desc": "New Mahatma Gandhi Series, Chocolate Brown color, numeral 10, Konark Sun Temple wheel motif",
        "ref_front": "inr_10_new_front.jpg",
        "ref_back": "inr_10_new_back.jpg",
        "bgr_color": (19, 69, 139),    # Chocolate Brown
    },
    "old10": {
        "name": "₹10 rupee banknote",
        "series": "Older Mahatma Gandhi Series",
        "denom": "10",
        "canonical_label": "10_rupee_note",
        "desc": "Older series, Orange-Violet color, numeral 10, Rhinoceros, Elephant, and Bengal Tiger motif",
        "ref_front": "inr_10_old_front.jpg",
        "ref_back": "inr_10_old_back.jpg",
        "bgr_color": (30, 105, 210),   # Orange-Violet
    },
    "new20": {
        "name": "₹20 rupee banknote",
        "series": "New Mahatma Gandhi Series",
        "denom": "20",
        "canonical_label": "20_rupee_note",
        "desc": "New Mahatma Gandhi Series, Greenish-Yellow color, numeral 20, Ellora Caves Kailash Temple motif",
        "ref_front": "inr_20_new_front.jpg",
        "ref_back": "inr_20_new_back.jpg",
        "bgr_color": (0, 215, 255),    # Greenish-Yellow
    },
    "old20": {
        "name": "₹20 rupee banknote",
        "series": "Older Mahatma Gandhi Series",
        "denom": "20",
        "canonical_label": "20_rupee_note",
        "desc": "Older series, Reddish-Orange color, numeral 20, Mount Harriet motif",
        "ref_front": "inr_20_old_front.jpg",
        "ref_back": "inr_20_old_back.jpg",
        "bgr_color": (0, 140, 255),    # Reddish-Orange
    },
    "new50": {
        "name": "₹50 rupee banknote",
        "series": "New Mahatma Gandhi Series",
        "denom": "50",
        "canonical_label": "50_rupee_note",
        "desc": "New Mahatma Gandhi Series, Fluorescent Cyan-Blue color, numeral 50, Hampi with Chariot motif",
        "ref_front": "inr_50_new_front.jpg",
        "ref_back": "inr_50_new_back.jpg",
        "bgr_color": (255, 191, 0),    # Fluorescent Cyan-Blue
    },
    "old50": {
        "name": "₹50 rupee banknote",
        "series": "Older Mahatma Gandhi Series",
        "denom": "50",
        "canonical_label": "50_rupee_note",
        "desc": "Older series, Violet-Pink color, numeral 50, Parliament of India motif",
        "ref_front": "inr_50_old_front.jpg",
        "ref_back": "inr_50_old_back.jpg",
        "bgr_color": (180, 105, 255),   # Violet-Pink
    },
    "old100": {
        "name": "₹100 rupee banknote",
        "series": "Older Mahatma Gandhi Series",
        "denom": "100",
        "canonical_label": "100_rupee_note",
        "desc": "Older series, Blue-Green color, numeral 100, Mount Kangchenjunga motif",
        "ref_front": "inr_100_old_front.jpg",
        "ref_back": "inr_100_old_back.jpg",
        "bgr_color": (200, 50, 180),   # Blue-Green
    },
    "new100": {
        "name": "₹100 rupee banknote",
        "series": "New Mahatma Gandhi Series",
        "denom": "100",
        "canonical_label": "100_rupee_note",
        "desc": "New Mahatma Gandhi Series, Lavender-Purple color, numeral 100, Rani ki Vav motif",
        "ref_front": "inr_100_new_front.png",
        "ref_back": "inr_100_new_back.png",
        "bgr_color": (220, 100, 220),  # Lavender
    },
    "new200": {
        "name": "₹200 rupee banknote",
        "series": "New Mahatma Gandhi Series",
        "denom": "200",
        "canonical_label": "200_rupee_note",
        "desc": "New Mahatma Gandhi Series, Bright Orange-Yellow color, numeral 200, Sanchi Stupa motif",
        "ref_front": "inr_200_new_front.jpg",
        "ref_back": "inr_200_new_back.jpg",
        "bgr_color": (0, 165, 255),    # Bright Orange-Yellow
    },
    "new500": {
        "name": "₹500 rupee banknote",
        "series": "New Mahatma Gandhi Series",
        "denom": "500",
        "canonical_label": "500_rupee_note",
        "desc": "New Mahatma Gandhi Series, Stone Grey color, numeral 500, Red Fort with Indian flag motif",
        "ref_front": "inr_500_new_front.jpg",
        "ref_back": "inr_500_new_back.jpg",
        "bgr_color": (128, 128, 128),  # Stone Grey
    },
    "new2000": {
        "name": "₹2000 rupee banknote",
        "series": "New Mahatma Gandhi Series",
        "denom": "2000",
        "canonical_label": "2000_rupee_note",
        "desc": "New Mahatma Gandhi Series, Magenta-Pink color, numeral 2000, Mangalyaan motif",
        "ref_front": "inr_2000_new_front.jpg",
        "ref_back": "inr_2000_new_back.jpg",
        "bgr_color": (180, 50, 255),   # Magenta
    },
}

# ============================ PROMPT BUILDERS ============================
def build_stage1_text_prompt(class_name, label_to_use):
    """Stage 1: Laser-focused text prompt for high-speed single-image detection."""
    info = CURRENCY_CLASSES.get(class_name)
    target_desc = f"Indian {info['name']} ({info['desc']})" if info else f"Indian {class_name} currency note"
    return f'''Locate and detect all {target_desc} in this image (even if folded, held by hand, partially visible, tilted, crumpled, or shadowed).
Respond ONLY with a valid JSON list in this exact format:
[{{"box_2d": [ymin, xmin, ymax, xmax], "label": "{label_to_use}"}}]
Coordinates are integers from 0 to 1000 (order: ymin, xmin, ymax, xmax).
If no {target_desc} is present, respond with: []'''

def build_stage2_ref_prompt(class_name, label_to_use):
    """Stage 2: Target-First visual reference prompt for difficult/missed cases."""
    info = CURRENCY_CLASSES.get(class_name)
    target_desc = f"Indian {info['name']} ({info['desc']})" if info else f"Indian {class_name} currency note"
    return f'''Locate and detect all {target_desc} in Image 1 (the target image to annotate).
Image 2 (front) and Image 3 (back) provide official visual references for this denomination.
Detect ONLY the visible boundaries of the banknote in Image 1 (even if folded, held by hand, partially visible, or shadowed). Do not include table, floor, or hands.
Respond ONLY with a valid JSON list in this exact format:
[{{"box_2d": [ymin, xmin, ymax, xmax], "label": "{label_to_use}"}}]
Coordinates are normalized integers from 0 to 1000 for Image 1.
If no {target_desc} is present in Image 1, respond with: []'''

# ============================ PARSING & VOC EXPORT ============================
_TOKENS_RE = re.compile(r"<start_of_turn>|<end_of_turn>|<pad>|</s>|<eos>|<bos>|<turn\|>")
_OBJ_RE = re.compile(
    r'\{\s*"box_2d"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]'
    r'\s*,\s*"label"\s*:\s*"([^"]*)"', re.DOTALL)
_OBJ_RE_LABEL_FIRST = re.compile(
    r'\{\s*"label"\s*:\s*"([^"]*)"\s*,\s*"box_2d"\s*:\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,'
    r'\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]', re.DOTALL)

NON_MONEY_RE = re.compile(r"\b(coin|person|hand|finger|fingers|table|desk|background|card|credit card|id card|wallet)\b", re.I)
_NEG_PHRASES = ("not find", "no bounding box", "not visible", "no currency", "no banknote", "none present", "cannot find", "no rupee")

def parse_detections(text):
    """Extract bounding boxes and raw labels from model response."""
    if not isinstance(text, str) or not text.strip():
        return [], False
    text = _TOKENS_RE.sub("", text).strip()
    if any(p in text.lower() for p in _NEG_PHRASES):
        return [], True
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        snippet = m.group(0)
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                out = [{"box_2d": [float(v) for v in d["box_2d"]], "label": str(d.get("label", ""))}
                       for d in data if isinstance(d, dict) and "box_2d" in d and len(d["box_2d"]) == 4]
                return out, True
        except json.JSONDecodeError:
            pass
    out = [{"box_2d": [float(a), float(b), float(c), float(d)], "label": lbl}
           for a, b, c, d, lbl in _OBJ_RE.findall(text)]
    out += [{"box_2d": [float(a), float(b), float(c), float(d)], "label": lbl}
            for lbl, a, b, c, d in _OBJ_RE_LABEL_FIRST.findall(text)]
    return out, bool(out)

def map_label(raw, expected_label):
    """Validate and map detected label to target label."""
    clean = re.sub(r"[-_]", " ", raw).lower().strip()
    if not clean or NON_MONEY_RE.fullmatch(clean):
        return None
    return expected_label

def _iou(a, b):
    ay1, ax1, ay2, ax2 = a["box_2d"]
    by1, bx1, by2, bx2 = b["box_2d"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    ab = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + ab - inter)

def nms(boxes, iou_thr=NMS_IOU):
    boxes = sorted(boxes, key=lambda b: (b["box_2d"][2] - b["box_2d"][0]) * (b["box_2d"][3] - b["box_2d"][1]), reverse=True)
    keep = []
    for b in boxes:
        if all(_iou(b, k) < iou_thr for k in keep):
            keep.append(b)
    return keep

def create_voc_xml(image_path, width, height, objects, output_xml_path, folder_name="."):
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = folder_name
    ET.SubElement(ann, "filename").text = os.path.basename(image_path)
    ET.SubElement(ann, "path").text = os.path.abspath(image_path)
    src = ET.SubElement(ann, "source")
    ET.SubElement(src, "database").text = "AutoAnnotated"
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(ann, "segmented").text = "0"
    for obj in objects:
        y1, x1, y2, x2 = (min(1000.0, max(0.0, float(c))) / 1000.0 for c in obj["box_2d"])
        xmin = min(max(1, round(x1 * width)), width - 1)
        xmax = min(max(1, round(x2 * width)), width - 1)
        ymin = min(max(1, round(y1 * height)), height - 1)
        ymax = min(max(1, round(y2 * height)), height - 1)
        if xmax <= xmin or ymax <= ymin:
            continue
        o = ET.SubElement(ann, "object")
        ET.SubElement(o, "name").text = obj["label"]
        ET.SubElement(o, "pose").text = "Unspecified"
        ET.SubElement(o, "truncated").text = "0"
        ET.SubElement(o, "difficult").text = "0"
        b = ET.SubElement(o, "bndbox")
        ET.SubElement(b, "xmin").text = str(xmin)
        ET.SubElement(b, "ymin").text = str(ymin)
        ET.SubElement(b, "xmax").text = str(xmax)
        ET.SubElement(b, "ymax").text = str(ymax)

    ET.indent(ann, space="    ")
    ET.ElementTree(ann).write(output_xml_path, encoding="utf-8", xml_declaration=True)

def save_preview(img_path, objects, out_path, color=(0, 255, 0)):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    for obj in objects:
        y1, x1, y2, x2 = (min(1000.0, max(0.0, float(c))) / 1000.0 for c in obj["box_2d"])
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        lbl = obj["label"]

        # Draw bounding box
        cv2.rectangle(img, p1, p2, color, 2)

        # Draw readable label banner
        text_size, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tb_x1, tb_y1 = p1[0], max(0, p1[1] - 22)
        tb_x2, tb_y2 = p1[0] + text_size[0] + 6, tb_y1 + 20
        cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), color, -1)
        cv2.putText(img, lbl, (tb_x1 + 3, tb_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)

# ============================ CASCADED PIPELINE ============================
def run_cascaded_labeler(
    dataset_dir=DEFAULT_DATASET_DIR,
    ref_dir=DEFAULT_REF_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    preview_dir=DEFAULT_PREVIEW_DIR,
    missed_dir=DEFAULT_MISSED_DIR,
    error_dir=DEFAULT_ERROR_DIR,
    report_path=DEFAULT_REPORT_PATH,
    raw_log_path=DEFAULT_RAW_LOG,
    num_per_class=None,
    target_classes=None,
    label_mode="denom",      # 'denom' -> '10_rupee_note', 'class' -> 'new10'
    mode="cascade",          # 'cascade', 'text-only', 'refs-only'
    skip_existing=False,
    preview_count_per_class=15
):
    """
    Two-stage auto-labeler:
      1. Run class-aware text prompt.
      2. If missed and in 'cascade' mode, automatically re-run with visual front & back references.
    """
    for d in (output_dir, preview_dir, missed_dir, error_dir):
        os.makedirs(d, exist_ok=True)

    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        return

    subdirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if target_classes:
        subdirs = [d for d in subdirs if d in target_classes]

    if not subdirs:
        print(f"No matching class subdirectories found in {dataset_dir}!")
        return

    print("=" * 70)
    print(" TWO-STAGE CASCADED INDIAN RUPEE AUTO-LABELER")
    print("=" * 70)
    print(f"Dataset root          : {dataset_dir}")
    print(f"Execution mode        : {mode.upper()}")
    print(f"References directory  : {ref_dir}")
    print(f"Classes found         : {', '.join(subdirs)} ({len(subdirs)} classes)")
    print(f"Images per class      : {num_per_class if num_per_class else 'ALL'}")
    print(f"Label mode            : {label_mode} (e.g. '{'10_rupee_note' if label_mode == 'denom' else 'new10'}')")
    print(f"Output XMLs           : {output_dir}/<class_name>/")
    print(f"Visual Previews       : {preview_dir}/<class_name>/")
    print("=" * 70)

    # Collect images
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    class_images = {}
    total_imgs = 0

    for cls in subdirs:
        cls_folder = os.path.join(dataset_dir, cls)
        imgs = sorted({p for e in exts for p in glob.glob(os.path.join(cls_folder, e))})

        if skip_existing:
            cls_out = os.path.join(output_dir, cls)
            existing = {Path(p).stem for p in glob.glob(os.path.join(cls_out, "*.xml"))}
            imgs = [p for p in imgs if Path(p).stem not in existing]

        if num_per_class and num_per_class < len(imgs):
            imgs = imgs[:num_per_class]

        class_images[cls] = imgs
        total_imgs += len(imgs)
        print(f"  [{cls:<8}] -> {len(imgs)} images queued")

    print(f"\nTotal images across all classes: {total_imgs}\n")
    if total_imgs == 0:
        print("No images to process.")
        return

    # Load Model & Processor
    print(f"Loading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        attn_implementation="sdpa"
    ).eval()

    torch.cuda.empty_cache()
    print(f"Model loaded. VRAM allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    overall_stats = {
        "processed": 0,
        "stage1_text_det": 0,
        "stage2_ref_det": 0,
        "total_detected": 0,
        "total_notes": 0,
        "missed": 0,
        "by_class": {}
    }
    rows = []
    rawlog = open(raw_log_path, "w")

    try:
        for cls, img_paths in class_images.items():
            if not img_paths:
                continue

            info = CURRENCY_CLASSES.get(cls, {})
            label_to_use = info.get("canonical_label", f"{cls}_rupee_note") if label_mode == "denom" else cls
            bgr_color = info.get("bgr_color", (0, 255, 0))

            # Pre-load front & back reference images
            ref_front_img = None
            ref_back_img = None
            has_refs = False

            if info.get("ref_front") and info.get("ref_back"):
                front_p = os.path.join(ref_dir, info["ref_front"])
                back_p = os.path.join(ref_dir, info["ref_back"])
                if os.path.exists(front_p) and os.path.exists(back_p):
                    try:
                        ref_front_img = Image.open(front_p).convert("RGB")
                        ref_back_img = Image.open(back_p).convert("RGB")
                        has_refs = True
                    except Exception as e:
                        print(f"  Warning: Could not open reference images for {cls}: {e}")

            # Pre-build Stage 1 Chat Template (Single-Image Text Prompt)
            stage1_prompt = build_stage1_text_prompt(cls, label_to_use)
            msg1 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": stage1_prompt}]}]
            chat_prompt_stage1 = processor.apply_chat_template(msg1, add_generation_prompt=True, tokenize=False)

            # Pre-build Stage 2 Chat Template (Target-First Multi-Image Reference Prompt)
            chat_prompt_stage2 = None
            if has_refs:
                stage2_prompt = build_stage2_ref_prompt(cls, label_to_use)
                msg2 = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Image 1 (Target image to detect note in):"},
                            {"type": "image"},
                            {"type": "text", "text": "Image 2 (Reference Front):"},
                            {"type": "image"},
                            {"type": "text", "text": "Image 3 (Reference Back):"},
                            {"type": "image"},
                            {"type": "text", "text": stage2_prompt}
                        ]
                    }
                ]
                chat_prompt_stage2 = processor.apply_chat_template(msg2, add_generation_prompt=True, tokenize=False)

            cls_out_dir = os.path.join(output_dir, cls)
            cls_preview_dir = os.path.join(preview_dir, cls)
            cls_missed_dir = os.path.join(missed_dir, cls)
            for d in (cls_out_dir, cls_preview_dir, cls_missed_dir):
                os.makedirs(d, exist_ok=True)

            cls_stats = {
                "count": len(img_paths),
                "stage1_det": 0,
                "stage2_det": 0,
                "total_det": 0,
                "notes": 0,
                "missed": 0,
                "errors": 0
            }
            previews_saved = 0

            pbar = tqdm(img_paths, desc=f"Class [{cls:<7}]", unit="img")
            for img_path in pbar:
                name = os.path.basename(img_path)
                base = os.path.splitext(name)[0]

                try:
                    image = Image.open(img_path).convert("RGB")
                    w, h = image.size
                except Exception as e:
                    print(f"  ERROR reading {name}: {e}")
                    shutil.copy(img_path, error_dir)
                    cls_stats["errors"] += 1
                    rows.append((cls, name, "unreadable", "none", 0, "", ""))
                    continue

                boxes = []
                method_used = "stage1_text"
                raw_text = ""

                # ---------------- STAGE 1: TEXT-ONLY RUN ----------------
                if mode in ("cascade", "text-only"):
                    inputs1 = processor(text=chat_prompt_stage1, images=image, return_tensors="pt").to(model.device)
                    with torch.inference_mode():
                        out1 = model.generate(**inputs1, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                    in_len1 = inputs1["input_ids"].shape[1]
                    raw_text = processor.decode(out1[0][in_len1:], skip_special_tokens=True).strip()

                    del inputs1, out1
                    torch.cuda.empty_cache()

                    detections1, _ = parse_detections(raw_text)
                    boxes = [{"box_2d": d["box_2d"], "label": lbl}
                             for d in detections1 if (lbl := map_label(d["label"], label_to_use))]
                    if APPLY_NMS and len(boxes) > 1:
                        boxes = nms(boxes)

                # ---------------- STAGE 2: VISUAL REFERENCE FALLBACK ----------------
                if (not boxes) and (mode in ("cascade", "refs-only")) and has_refs and chat_prompt_stage2:
                    method_used = "stage2_refs"
                    # Pass target image FIRST, followed by references
                    inputs2 = processor(text=chat_prompt_stage2, images=[image, ref_front_img, ref_back_img], return_tensors="pt").to(model.device)
                    with torch.inference_mode():
                        out2 = model.generate(**inputs2, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                    in_len2 = inputs2["input_ids"].shape[1]
                    raw_text = processor.decode(out2[0][in_len2:], skip_special_tokens=True).strip()

                    del inputs2, out2
                    torch.cuda.empty_cache()

                    detections2, _ = parse_detections(raw_text)
                    boxes = [{"box_2d": d["box_2d"], "label": lbl}
                             for d in detections2 if (lbl := map_label(d["label"], label_to_use))]
                    if APPLY_NMS and len(boxes) > 1:
                        boxes = nms(boxes)

                # ---------------- LOGGING & EXPORT ----------------
                rawlog.write(json.dumps({"class": cls, "image": name, "stage": method_used, "raw": raw_text}) + "\n")
                rawlog.flush()

                if boxes:
                    create_voc_xml(
                        image_path=img_path,
                        width=w,
                        height=h,
                        objects=boxes,
                        output_xml_path=os.path.join(cls_out_dir, f"{base}.xml"),
                        folder_name=cls
                    )
                    if method_used == "stage1_text":
                        cls_stats["stage1_det"] += 1
                    else:
                        cls_stats["stage2_det"] += 1

                    cls_stats["total_det"] += 1
                    cls_stats["notes"] += len(boxes)
                    labels_str = ";".join(b["label"] for b in boxes)
                    rows.append((cls, name, "annotated", method_used, len(boxes), labels_str, raw_text))

                    if previews_saved < preview_count_per_class:
                        save_preview(
                            img_path,
                            boxes,
                            os.path.join(cls_preview_dir, f"{base}.jpg"),
                            color=bgr_color
                        )
                        previews_saved += 1
                else:
                    shutil.copy(img_path, cls_missed_dir)
                    cls_stats["missed"] += 1
                    rows.append((cls, name, "not_detected", method_used, 0, "", raw_text))

                pct = 100 * cls_stats["total_det"] / max(cls_stats["total_det"] + cls_stats["missed"], 1)
                pbar.set_postfix_str(f"det: {cls_stats['total_det']}/{len(img_paths)} ({pct:.0f}%) | s1:{cls_stats['stage1_det']} s2:{cls_stats['stage2_det']}")

            pbar.close()
            overall_stats["by_class"][cls] = cls_stats
            overall_stats["processed"] += cls_stats["count"]
            overall_stats["stage1_text_det"] += cls_stats["stage1_det"]
            overall_stats["stage2_ref_det"] += cls_stats["stage2_det"]
            overall_stats["total_detected"] += cls_stats["total_det"]
            overall_stats["total_notes"] += cls_stats["notes"]
            overall_stats["missed"] += cls_stats["missed"]

    finally:
        rawlog.close()

    # Save summary report CSV
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "image", "status", "stage_used", "num_boxes", "detected_labels", "raw_output"])
        writer.writerows(rows)

    # Print summary table
    print("\n" + "=" * 80)
    print(" CASCADED DETECTION SUMMARY")
    print("=" * 80)
    print(f"{'Class':<10} | {'Images':<8} | {'Stage 1 (Text)':<16} | {'Stage 2 (Refs)':<16} | {'Total Det':<10} | {'Rate':<8}")
    print("-" * 80)

    for cls, st in overall_stats["by_class"].items():
        rate = 100 * st["total_det"] / max(st["count"], 1)
        print(f"{cls:<10} | {st['count']:<8} | {st['stage1_det']:<16} | {st['stage2_det']:<16} | {st['total_det']:<10} | {rate:>5.1f}%")

    total_proc = overall_stats["processed"]
    total_det = overall_stats["total_detected"]
    total_rate = 100 * total_det / max(total_proc, 1)

    print("-" * 80)
    print(f"{'OVERALL':<10} | {total_proc:<8} | {overall_stats['stage1_text_det']:<16} | {overall_stats['stage2_ref_det']:<16} | {total_det:<10} | {total_rate:>5.1f}%")
    print("=" * 80)
    print(f"\nPascal VOC XMLs saved to : {output_dir}/<class_name>/")
    print(f"Visual previews saved to : {preview_dir}/<class_name>/")
    print(f"Detailed CSV report      : {report_path}")
    print(f"Raw outputs log          : {raw_log_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascaded Class-Aware Indian Rupee Banknote Detector")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="Path to dataset directory")
    parser.add_argument("--ref-dir", type=str, default=DEFAULT_REF_DIR, help="Path to currency_references directory")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save VOC XMLs")
    parser.add_argument("--preview-dir", type=str, default=DEFAULT_PREVIEW_DIR, help="Directory to save visual previews")
    parser.add_argument("--missed-dir", type=str, default=DEFAULT_MISSED_DIR, help="Directory to copy missed images")
    parser.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH, help="Path to summary CSV report")
    parser.add_argument("--raw-log-path", type=str, default=DEFAULT_RAW_LOG, help="Path to raw JSONL log")
    parser.add_argument("--mode", type=str, default="cascade", choices=["cascade", "text-only", "refs-only"],
                        help="'cascade' (Stage 1 text -> Stage 2 visual fallback), 'text-only', or 'refs-only'")
    parser.add_argument("--images-per-class", type=int, default=0, help="Images per class (0 for all images)")
    parser.add_argument("--classes", nargs="+", default=None, help="Specific classes to run (e.g. --classes new10 old10)")
    parser.add_argument("--label-mode", type=str, default="denom", choices=["denom", "class"],
                        help="'denom' for 10_rupee_note, 'class' for new10/old10")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="Skip images with existing XMLs")
    parser.add_argument("--preview-count-per-class", type=int, default=15, help="Previews to save per class")

    args = parser.parse_args()
    num_per_class = None if args.images_per_class <= 0 else args.images_per_class

    run_cascaded_labeler(
        dataset_dir=args.dataset_dir,
        ref_dir=args.ref_dir,
        output_dir=args.output_dir,
        preview_dir=args.preview_dir,
        missed_dir=args.missed_dir,
        report_path=args.report_path,
        raw_log_path=args.raw_log_path,
        num_per_class=num_per_class,
        target_classes=args.classes,
        label_mode=args.label_mode,
        mode=args.mode,
        skip_existing=args.skip_existing,
        preview_count_per_class=args.preview_count_per_class
    )
