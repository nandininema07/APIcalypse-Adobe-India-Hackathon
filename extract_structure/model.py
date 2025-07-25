#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import easyocr
import re
import numpy as np
import json
import os

from preprocessing import *
from preprocessing import run_pdf_to_blocks

from compress_pdf import compress_pdf

# In[6]:

def group_blocks_same_line(blocks, y_tolerance=5):
    blocks = sorted(blocks, key=lambda b: (b.get('page', 0), b['box'][1], b['box'][0]))
    grouped = []
    current = None

    for b in blocks:
        if not b.get('text', '').strip():
            continue

        if current is None:
            current = b
        else:
            same_page = current.get('page') == b.get('page')
            same_line = abs(current['box'][1] - b['box'][1]) <= y_tolerance

            if same_page and same_line:
                current['text'] += ' ' + b['text'].strip()
                current['box'][0] = min(current['box'][0], b['box'][0])
                current['box'][2] = max(current['box'][2], b['box'][2])
                current['box'][3] = max(current['box'][3], b['box'][3])
            else:
                grouped.append(current)
                current = b

    if current:
        grouped.append(current)

    return grouped

def build_tree_outline(predictions):
    # Sort by predicted order
    predictions = sorted(predictions, key=lambda x: (x['order'], x['id']))

    id_to_node = {}
    for item in predictions:
        # Filter out weak headings (too short, lowercase, etc.)
        cleaned_text = item['text'].strip()
        if len(cleaned_text.split()) < 2 or cleaned_text.islower():
            continue

        id_to_node[item['id']] = {
            "id": item['id'],
            "text": cleaned_text,
            "level": f"H{min(item['order']+1, 3)}",
            "page": int(item.get('page', 0)) + 1,
            "children": [],
            "parent_id": item['parent_id']
        }

    root_nodes = []
    for node in id_to_node.values():
        parent_id = node.pop("parent_id")
        if parent_id != 0 and parent_id in id_to_node:
            id_to_node[parent_id]['children'].append(node)
        else:
            root_nodes.append(node)

    return root_nodes


def build_tree_outline(predictions):
    predictions = sorted(predictions, key=lambda x: (x['order'], x['id']))
    
    id_to_node = {}
    for item in predictions:
        id_to_node[item['id']] = {
            "id": item['id'],
            "text": item['text'].strip(),
            "level": f"H{min(item['order']+1, 3)}",
            "page": int(item.get('page', 0)) + 1,
            "children": [],
            "parent_id": item['parent_id']
        }

    root_nodes = []
    for node in id_to_node.values():
        parent_id = node.pop("parent_id")
        if parent_id != 0 and parent_id in id_to_node:
            id_to_node[parent_id]['children'].append(node)
        else:
            root_nodes.append(node)
    
    return root_nodes


def extract_paragraphs_with_ocr(pdf_path):
    structured_blocks = []
    page_sizes = {}
    order = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, 0):
            page_sizes[str(page_idx)] = {"width": page.width, "height": page.height}
            lines = list(page.extract_text_lines())
            if lines and any(line.get("text", "").strip() for line in lines) and len(lines) >= 5:
                for line in lines:
                    box = [line["x0"], line["top"], line["x1"], line["bottom"]]
                    if any(coord < 0 or coord > max(page.width, page.height) for coord in box):
                        continue
                    block = {
                        "box": box,
                        "text": line["text"] or "[EMPTY]",
                        "page": page_idx,
                        "id": order,
                        "order": order
                    }
                    structured_blocks.append(block)
                    order += 1
            else:
                try:
                    pil_img = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(pil_img, lang='eng')
                except Exception as e:
                    print(f"OCR failed for page {page_idx} in {pdf_path}: {e}")
                    ocr_text = "[EMPTY]"
                block = {
                    "box": [0, 0, page.width, page.height],
                    "text": ocr_text or "[EMPTY]",
                    "page": page_idx,
                    "id": order,
                    "order": order
                }
                structured_blocks.append(block)
                order += 1

    return {
        # "pages": page_sizes,
        "contents": structured_blocks
    }

test_pdf_dir = '../input'
test_output_dir = './json_input'
compressed_dir = os.path.join(os.path.dirname(__file__), "compressed_inputs")
os.makedirs(compressed_dir, exist_ok=True)

for fname in os.listdir(test_pdf_dir):
    if fname.lower().endswith('.pdf'):
        in_pdf = os.path.join(test_pdf_dir, fname)
        input_size_kb = os.path.getsize(in_pdf) // 1024
        if input_size_kb > 1000:
            compressed_pdf = os.path.join(compressed_dir, f"compressed_{fname}")
            compressed = compress_pdf(in_pdf, compressed_pdf, max_size_kb=1000, quality=80)
            if compressed:
                print(f"Compressed {fname} to {compressed_pdf}")
                in_pdf = compressed_pdf  # Use compressed file for further processing
            else:
                print(f"No compression needed for {fname}")
        out_json = os.path.join(test_output_dir, os.path.splitext(fname)[0] + ".json")
        doc = extract_paragraphs_with_ocr(in_pdf)
        os.makedirs(os.path.dirname(out_json), exist_ok=True)  # <-- Add this line
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"Processed: {fname} -> {out_json}")

# def clean_text(text):
#     # Remove repeated characters (e.g., 'RRRR' -> 'R'), excessive whitespace, and non-printables
#     text = re.sub(r'(.)\1{2,}', r'\1', text)  # Collapse 3+ repeated chars
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip()
#     return text

def clean_text(text):
    # Remove repeated characters, excessive whitespace, and non-printables
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # Collapse 3+ repeated chars
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# def is_heading_candidate(text):
#     # Stricter rules: starts with number, all caps, or short (not full sentence)
#     text = text.strip()
#     if len(text) < 3 or len(text) > 100:
#         return False
#     if re.match(r'^[A-Z][A-Z\s\d:,-]+$', text) and len(text.split()) <= 10:
#         return True
#     if re.match(r'^(\d+(\.\d+)*)(\.|\))?\s+.+', text):
#         return True
#     if len(text.split()) <= 8 and text[-1] not in '.!?':
#         return True
#     return False

def is_heading_candidate(text):
    # Clean and check for heading-like patterns
    text = text.strip()
    if len(text) < 3 or len(text) > 100:
        return False

    # Avoid common non-heading phrases
    lower_text = text.lower()
    if re.search(r'\b(later than|for the|to the|as they|such as|used by)\b', lower_text):
        return False

    # Reject bullet points and overly punctuated lines
    if text.startswith('-') or text.endswith((',', ';')):
        return False

    # Reject long sentences or lowercase-only lines
    if len(text.split()) > 12 or text.islower():
        return False

    # Strong candidates: all caps, numbered, short phrases
    if re.match(r'^[A-Z][A-Z\s\d:,-]+$', text) and len(text.split()) <= 10:
        return True
    if re.match(r'^(\d+(\.\d+)*)(\.|\))?\s+.+', text):
        return True
    if len(text.split()) <= 8 and text[-1] not in '.!?':
        return True

    return False


def get_heading_level_from_text(text):
    # Only allow up to H3
    match = re.match(r'^(\d+)(\.\d+)?(\.\d+)?(\.|\))', text.strip())
    if match:
        level = text.strip().count('.') + 1
        return f'h{min(level, 3)}'  # Cap at h3
    return None

def get_heading_level_from_indent(box, all_boxes):
    x0s = [b[0] for b in all_boxes]
    unique_x0s = sorted(set(x0s))
    clusters = []
    tol = 10  # pixels
    for x in unique_x0s:
        found = False
        for c in clusters:
            if abs(c[0] - x) < tol:
                c.append(x)
                found = True
                break
        if not found:
            clusters.append([x])
    for idx, cluster in enumerate(sorted(clusters, key=lambda c: min(c))):
        if abs(box[0] - np.mean(cluster)) < tol:
            return f'h{min(idx+1, 3)}'  # Cap at h3
    return None

def assign_heading_levels(blocks):
    all_boxes = [b['box'] for b in blocks if b['label'] in ['title', 'section-title'] and 'box' in b]
    for block in blocks:
        block['text'] = clean_text(block['text'])
        if block['label'] == 'title':
            block['heading_level'] = 'title'
            continue
        if block['label'] == 'section-title' and is_heading_candidate(block['text']):
            level = get_heading_level_from_text(block['text'])
            if level:
                block['heading_level'] = level
                continue
            if 'box' in block:
                level = get_heading_level_from_indent(block['box'], all_boxes)
                if level:
                    block['heading_level'] = level
                    continue
            if 'order' in block and isinstance(block['order'], int):
                block['heading_level'] = f'h{min(block["order"]+1, 3)}'
                continue
            block['heading_level'] = 'h1'
        else:
            block['heading_level'] = 'other'
    return blocks

def extract_title(blocks):
    # Pick the largest, boldest, or top-most text block on the first page
    first_page_blocks = [b for b in blocks if b.get('page', 0) == 0 and b['label'] == 'title']
    if not first_page_blocks:
        first_page_blocks = [b for b in blocks if b.get('page', 0) == 0]
    # Prefer block with largest box height (as proxy for font size)
    if first_page_blocks:
        best = max(first_page_blocks, key=lambda b: (b['box'][3] - b['box'][1]))
        return clean_text(best['text'])
    # Fallback: first block
    if blocks:
        return clean_text(blocks[0]['text'])
    return ""

# def convert_to_outline_format(heading_blocks):
#     title = extract_title(heading_blocks)
#     outline = []
#     for block in heading_blocks:
#         level = block.get('heading_level', '')
#         if level.startswith('h') and level != 'other':
#             outline.append({
#                 "level": level.upper(),  # e.g., "h1" -> "H1"
#                 "text": block['text'].strip(),
#                 "page": int(block.get('page', 0)) + 1  # +1 if your pages are 0-indexed
#             })
#     return {
#         "title": title,
#         "outline": outline
#     }

def convert_to_outline_format(heading_blocks):
    title = extract_title(heading_blocks)
    outline = []
    seen = set()
    for block in heading_blocks:
        level = block.get('heading_level', '')
        text = block['text'].strip()
        page = int(block.get('page', 0)) + 1
        key = (level, text.lower(), page)
        if level.startswith('h') and level != 'other' and key not in seen:
            outline.append({
                "level": level.upper(),
                "text": text,
                "page": page
            })
            seen.add(key)
    return {
        "title": title,
        "outline": outline
    }


def main():
    start_time = time.time()  # Start the timer
    print("=== DocIENet Model Inference for PDF Input ===")
    input_dir = "./json_input"
    output_dir = "../output"
    predictions_dir = "./model predictions"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json') and not f.endswith('_predictions.json')]

    if not input_files:
        print("No JSON files found in ./json_input. Please run preprocess_pdf.py.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained('./pretrained_models_bert_tiny')
    except Exception as e:
        print(f"Failed to load tokenizer from ./pretrained_models_bert_tiny: {e}")
        return

    device = torch.device('cpu')
    print(f"\nUsing device: {device}")

    try:
        inference_model = FastInferenceModel("./models")
    except Exception as e:
        print(f"Failed to initialize FastInferenceModel: {e}")
        return

    for input_file in input_files:
        print(f"Testing inference on {input_file}")
        try:
            results = inference_model.predict(input_file)
            # Filter predictions with confidence > 0.8
            filtered_results = [pred for pred in results if pred['confidence'] > 0.7]
            output_file = os.path.join(predictions_dir, os.path.basename(input_file).replace('.json', '_predictions.json'))
            if filtered_results:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_results, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(filtered_results)} predictions to {output_file}")
                # --- Add heading level assignment and save ---
                # Load the original input blocks
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_json = json.load(f)
                # --- GROUP FRAGMENTED BLOCKS ---
                grouped_blocks = group_blocks_same_line(input_json.get('contents', []))
                input_blocks = {block['id']: block for block in grouped_blocks}

                # Merge box and page into predictions
                for pred in filtered_results:
                    input_block = input_blocks.get(pred['id'])
                    if input_block:
                        pred['box'] = input_block.get('box', [0, 0, 0, 0])
                        pred['page'] = input_block.get('page', 0)
                    else:
                        pred['box'] = [0, 0, 0, 0]
                        pred['page'] = 0

                heading_results = assign_heading_levels(filtered_results)
                heading_file = os.path.join(predictions_dir, os.path.basename(input_file).replace('.json', '_headings.json'))
                with open(heading_file, 'w', encoding='utf-8') as f:
                    json.dump(heading_results, f, indent=2, ensure_ascii=False)
                print(f"Saved heading levels to {heading_file}")
                # --- Convert to outline format and save ---
                outline_json = convert_to_outline_format(heading_results)
                outline_file = os.path.join(output_dir, os.path.basename(input_file).replace('.json', '.json'))
                with open(outline_file, 'w', encoding='utf-8') as f:
                    json.dump(outline_json, f, indent=2, ensure_ascii=False)
                print(f"Saved outline to {outline_file}")

                # Save tree-style outline to model predictions folder
                tree_outline = build_tree_outline(heading_results)
                tree_file = os.path.join(predictions_dir, os.path.basename(input_file).replace('.json', '_outline_tree.json'))
                with open(tree_file, 'w', encoding='utf-8') as f:
                    json.dump(tree_outline, f, indent=2, ensure_ascii=False)
                print(f"Saved tree-style outline to {tree_file}")
                
            else:
                print(f"No predictions with confidence > 0.8 for {input_file}. Skipping output.")
            print("\nPredictions:")
            for pred in results:
                print(f"ID: {pred['id']}")
                print(f"  Pred label: {pred['label']}")
                print(f"  Pred order: {pred['order']}")
                print(f"  Pred parent_id: {pred['parent_id']}")
                print(f"  Text: {pred['text']}")
                print(f"  Confidence: {pred['confidence']:.4f}")
                print("---")
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
    
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"\nTotal time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()


# In[ ]:




