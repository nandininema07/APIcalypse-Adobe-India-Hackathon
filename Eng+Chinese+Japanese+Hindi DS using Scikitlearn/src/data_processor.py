# src/data_processor.py (Updates for DocHieNet processing and OCR language selection)

import os
import sys
import json
from typing import List, Dict, Any
import random
import re 
from fuzzywuzzy import fuzz 
import fitz

# --- Fix for ModuleNotFoundError: Add project root to sys.path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.insert(0, project_root)
# --- End of ModuleNotFoundError Fix ---

from src.pdf_parser import extract_page_elements, get_pdf_dimensions
from src.feature_extractor import group_elements_into_lines, group_lines_into_blocks, extract_features_from_blocks, encode_features_for_fasttext # Make sure to import THRESHOLD_INDENT if used in data_processor.py

# --- Configuration ---
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed', 'heading_classification')
SYNTHETIC_DATA_DIR = os.path.join(project_root, 'data', 'synthetic')

TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train.txt')
VAL_FILE = os.path.join(PROCESSED_DATA_DIR, 'val.txt')
TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'test.txt')

# --- IoU (Intersection over Union) helper function ---
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


# --- Process Data from DocHieNet & HJDataset Annotations (Conceptual Implementations) ---

# Mapping from DocHieNet's internal labels to our standard labels
DOC_HIENET_LABEL_MAP = {
    "title": "TITLE",
    "header": "H1", # Will be refined using 'level' from annotations
    "paragraph": "BODY_TEXT",
    "list": "BODY_TEXT", # Simplifying list items as body text for now for the classifier
    "caption": "BODY_TEXT",
    "table": "BODY_TEXT",
    "page_number": "BODY_TEXT",
    "figure": "BODY_TEXT", 
}

def process_doc_hienet(base_dir: str) -> List[str]:
    """
    Processes DocHieNet dataset to extract text blocks with ground-truth labels.
    Handles image files (JPG/PNG/PDF) using OCR fallback, iterating through pages.
    Correctly handles images organized in doc_id subfolders.
    """
    fasttext_lines = []
    
    labels_dir = os.path.join(base_dir, 'labels')
    
    # --- Determine the correct image base directory (images/ or hres_images/) ---
    img_base_folder = ""
    img_dir_option1 = os.path.join(base_dir, 'images') 
    img_dir_option2 = os.path.join(base_dir, 'hres_images') 
    
    # Check if dir exists AND is not empty (contains doc_id subfolders)
    if os.path.exists(img_dir_option1) and os.listdir(img_dir_option1): 
        img_base_folder = img_dir_option1
    elif os.path.exists(img_dir_option2) and os.listdir(img_dir_option2): 
        img_base_folder = img_dir_option2
    
    if not os.path.exists(labels_dir):
        print(f"WARNING: DocHieNet labels directory not found: {labels_dir}. Skipping DocHieNet processing.")
        return []
    if not img_base_folder:
        print(f"ERROR: DocHieNet image base directory not found or empty in '{img_dir_option1}' or '{img_dir_option2}'. Skipping DocHieNet processing.")
        return []

    print(f"DEBUG: Processing DocHieNet from {base_dir}. Using image base dir: {img_base_folder}")
    
    train_test_split_path = os.path.join(base_dir, 'train_test_split.json')
    if not os.path.exists(train_test_split_path):
        print(f"WARNING: DocHieNet train_test_split.json not found at {train_test_split_path}. Listing all JSONs in labels/ directly.")
        all_doc_ids_to_process = [f.replace('.json', '') for f in os.listdir(labels_dir) if f.endswith('.json')]
    else:
        with open(train_test_split_path, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        all_doc_ids_to_process = split_info.get('train', []) + split_info.get('test', []) # Combine train and test IDs

    print(f"DEBUG: Found {len(all_doc_ids_to_process)} DocHieNet documents to attempt processing.")

    processed_doc_count = 0
    MAX_DOCS_TO_PROCESS_DOC_HIENET = 50 # Limit for faster debugging/initial training

    for doc_id in all_doc_ids_to_process:
        if processed_doc_count >= MAX_DOCS_TO_PROCESS_DOC_HIENET:
            print(f"DEBUG: Reached MAX_DOCS_TO_PROCESS_DOC_HIENET ({MAX_DOCS_TO_PROCESS_DOC_HIENET}). Stopping DocHieNet processing for this run.")
            break 

        label_filepath = os.path.join(labels_dir, f"{doc_id}.json") 
        doc_specific_image_folder = os.path.join(img_base_folder, doc_id) # THIS IS THE CRITICAL CHANGE

        if not os.path.exists(label_filepath):
            # print(f"DEBUG: DocHieNet doc {doc_id}: Label JSON not found at {label_filepath}. Skipping.")
            continue 

        if not os.path.exists(doc_specific_image_folder) or not os.listdir(doc_specific_image_folder):
            # print(f"DEBUG: DocHieNet doc {doc_id}: Image folder for pages not found or empty: {doc_specific_image_folder}. Skipping.")
            continue 

        try:
            with open(label_filepath, 'r', encoding='utf-8') as f:
                doc_annotations = json.load(f)
        except Exception as e:
            print(f"WARNING: Could not load DocHieNet annotation {label_filepath}: {e}. Skipping document.")
            continue
        
        # --- NEW: Get all page image paths for this document based on annotations ---
        doc_page_image_paths = []
        # DocHieNet JSON 'pages' key contains width/height for "page1", "page2" etc.
        # This tells us how many pages and their dimensions.
        if doc_annotations.get('pages'):
            for page_key in doc_annotations['pages'].keys(): # Iterate through "page1", "page2"
                p_num_str = page_key.replace('page', '')
                try:
                    p_num_1_idx = int(p_num_str) # Get 1-indexed page number
                except ValueError:
                    continue # Skip if key is not like 'pageX'

                page_image_path = ""
                # Check for .jpg, .png, .pdf in that order
                if os.path.exists(os.path.join(doc_specific_image_folder, f"page{p_num_1_idx}.jpg")):
                    page_image_path = os.path.join(doc_specific_image_folder, f"page{p_num_1_idx}.jpg")
                elif os.path.exists(os.path.join(doc_specific_image_folder, f"page{p_num_1_idx}.png")):
                    page_image_path = os.path.join(doc_specific_image_folder, f"page{p_num_1_idx}.png")
                elif os.path.exists(os.path.join(doc_specific_image_folder, f"page{p_num_1_idx}.pdf")): 
                    page_image_path = os.path.join(doc_specific_image_folder, f"page{p_num_1_idx}.pdf")
                
                if page_image_path:
                    doc_page_image_paths.append(page_image_path)
        
        if not doc_page_image_paths:
            # print(f"DEBUG: DocHieNet doc {doc_id}: No page images found in {doc_specific_image_folder}. Skipping document.")
            continue


        # --- Extract elements from all pages of this document via OCR/words ---
        doc_all_parsed_elements = [] 
        ocr_lang_for_doc = "eng" # Default OCR language for DocHieNet (refine with en_zh_split.json for Chinese)

        for page_img_path in doc_page_image_paths:
            page_w, page_h = get_pdf_dimensions(page_img_path) 
            parsed_elements_for_page = extract_page_elements(page_img_path, lang=ocr_lang_for_doc) 
            doc_all_parsed_elements.extend(parsed_elements_for_page)

        if not doc_all_parsed_elements:
            # print(f"DEBUG: DocHieNet doc {doc_id}: No elements extracted from any page via OCR/words. Skipping document.")
            continue
        
        # Now, group all elements from this document into lines and blocks
        # Use the dimensions from the first page's image for grouping logic.
        first_page_img_path = doc_page_image_paths[0]
        doc_w, doc_h = get_pdf_dimensions(first_page_img_path) 

        lines_of_words = group_elements_into_lines(doc_all_parsed_elements, doc_w, doc_h)
        extracted_blocks = group_lines_into_blocks(lines_of_words, doc_w, doc_h)
        final_extracted_blocks_with_features = extract_features_from_blocks(extracted_blocks, doc_w, doc_h)

        if not final_extracted_blocks_with_features:
            # print(f"DEBUG: DocHieNet doc {doc_id}: No feature blocks formed from extracted words for entire document. Skipping.")
            continue

        # --- Match extracted blocks with DocHieNet annotations and assign labels ---
        # DocHieNet's JSON has a 'contents' key (the list of all annotated blocks)
        
        gt_annotated_elements = [] 
        for anno_block in doc_annotations.get('contents', []): 
            if anno_block.get('text') and anno_block.get('label'):
                gt_annotated_elements.append({
                    'text': anno_block['text'].replace('\n', ' ').strip(),
                    'bbox': anno_block['box'], # DocHieNet uses 'box' not 'bbox'
                    'label_orig': anno_block['label'], 
                    'level_num': anno_block.get('level'), 
                    'page_num': anno_block['page'] - 1 # Convert 1-indexed to 0-indexed
                })
        
        matched_parser_block_indices = set()
        
        for gt_anno_elem in gt_annotated_elements:
            gt_text_norm = gt_anno_elem['text'].lower()
            gt_bbox = gt_anno_elem['bbox']
            gt_label_orig = gt_anno_elem['label_orig']
            gt_level_num = gt_anno_elem['level_num'] 
            gt_page_num = gt_anno_elem['page_num']

            if gt_label_orig in ["footer", "page-number", "figure", "table", "caption", "equation"]: # Filter irrelevant labels
                continue

            mapped_label = DOC_HIENET_LABEL_MAP.get(gt_label_orig, "BODY_TEXT")
            # If original label is 'section-title', classify as H1.
            # If original label is 'header', use its 'level_num' from annotation.
            if gt_label_orig == "section-title":
                mapped_label = 'H1' # Simplified: Treat all DocHieNet section-titles as H1 for training.
            elif gt_label_orig == "header" and gt_level_num is not None:
                if gt_level_num == 1: mapped_label = 'H1'
                elif gt_level_num == 2: mapped_label = 'H2'
                elif gt_level_num == 3: mapped_label = 'H3'
                elif gt_level_num > 3: mapped_label = 'H3' 
                else: mapped_label = 'H1' 

            best_match_idx = -1
            max_iou = 0.0
            max_text_sim = 0.0

            for i, parsed_block in enumerate(final_extracted_blocks_with_features):
                if i in matched_parser_block_indices:
                    continue
                if parsed_block['page_num'] != gt_page_num:
                    continue
                
                current_iou = calculate_iou(gt_bbox, parsed_block['bbox'])
                current_text_sim = fuzz.token_sort_ratio(gt_text_norm, parsed_block['text'].lower()) / 100.0

                IOU_MATCH_THRESHOLD = 0.5 # Lowered IoU due to OCR bbox imprecision
                TEXT_SIM_MATCH_THRESHOLD = 0.65 # Lowered text sim due to OCR errors

                if (current_iou >= IOU_MATCH_THRESHOLD and current_text_sim >= TEXT_SIM_MATCH_THRESHOLD):
                    if current_iou > max_iou or (current_iou == max_iou and current_text_sim > max_text_sim):
                        max_iou = current_iou
                        max_text_sim = current_text_sim
                        best_match_idx = i
                elif current_text_sim >= 0.85: # If very high text sim (85%+) regardless of IoU (OCR perfect, bbox bad, or fragmentation)
                    if current_text_sim > max_text_sim: # Still prioritize best text sim
                        max_iou = current_iou # Record IoU for debug
                        max_text_sim = current_text_sim
                        best_match_idx = i
            
            if best_match_idx != -1:
                matched_parser_block_indices.add(best_match_idx)
                matched_block = final_extracted_blocks_with_features[best_match_idx]
                
                fasttext_input = encode_features_for_fasttext(matched_block['text'], matched_block)
                fasttext_lines.append(f"__label__{mapped_label} {fasttext_input}")
            # else: print(f"WARNING: DocHieNet: Could not match GT '{gt_text_norm}' ({gt_label_orig}, level {gt_level_num}) on page {gt_page_num} (IoU: {max_iou:.2f}, Sim: {max_text_sim:.2f}).")
        
        for i, block in enumerate(final_extracted_blocks_with_features):
            if i not in matched_parser_block_indices:
                if len(block['text'].strip()) > 5 and not re.match(r'^\s*\d+[\.-]?\s*$', block['text'].strip()):
                    fasttext_input = encode_features_for_fasttext(block['text'], block)
                    fasttext_lines.append(f"__label__BODY_TEXT {fasttext_input}")
        
        processed_doc_count += 1
        if processed_doc_count % 10 == 0:
            print(f"DEBUG: Processed {processed_doc_count} DocHieNet documents. Current total lines: {len(fasttext_lines)}")

    print(f"DEBUG: DocHieNet processing generated {len(fasttext_lines)} lines from {processed_doc_count} documents.")
    return fasttext_lines

def process_hj_dataset(base_dir: str) -> List[str]:
    """
    Processes HJDataset to extract text blocks with ground-truth labels.
    Requires deep understanding of HJDataset's JSON structure.
    Currently a placeholder returning dummy data.
    """
    fasttext_lines = []
    annotations_dir = os.path.join(base_dir, 'annotations')
    
    if not os.path.exists(annotations_dir):
        print(f"WARNING: HJDataset annotations directory not found: {annotations_dir}. Skipping actual processing.")
        return []

    print(f"DEBUG: Processing HJDataset from {base_dir} (returning dummy data for now).")
    dummy_elements_with_labels = [
        {'text': "日本語のタイトル例", 'bbox': [100,50,400,80], 'page_num': 0, 'font_size': 30, 'is_bold': True, 'is_italic': False, 'is_centered': True, 'y_spacing_relative': 0.1, 'x_indentation': 0.0, 'is_list_item': False, 'gt_label': 'TITLE', 'is_all_caps': False},
        {'text': "第一章：概要", 'bbox': [100,100,300,120], 'page_num': 0, 'font_size': 22, 'is_bold': True, 'is_italic': False, 'is_centered': False, 'y_spacing_relative': 0.05, 'x_indentation': 0.05, 'is_list_item': False, 'gt_label': 'H1', 'is_all_caps': False},
        {'text': "これは本文のサンプルです。", 'bbox': [100,130,500,150], 'page_num': 0, 'font_size': 12, 'is_bold': False, 'is_italic': False, 'is_centered': False, 'y_spacing_relative': 0.01, 'x_indentation': 0.05, 'is_list_item': False, 'gt_label': 'BODY_TEXT', 'is_all_caps': False},
    ]
    
    for block_data in dummy_elements_with_labels:
        fasttext_input = encode_features_for_fasttext(block_data['text'], block_data)
        fasttext_lines.append(f"__label__{block_data['gt_label']} {fasttext_input}")
    return fasttext_lines


# --- Main Data Processing Orchestration ---
def prepare_all_data():
    """Orchestrates the entire data preparation process."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) # Ensure output directory exists

    all_train_lines = []
    all_val_lines = []
    all_test_lines = []

    # 1. Process Synthetic Hindi Data (Already working well)
    synthetic_hi_path = os.path.join(SYNTHETIC_DATA_DIR, 'synthetic_hi.txt')
    if os.path.exists(synthetic_hi_path):
        with open(synthetic_hi_path, 'r', encoding='utf-8') as f:
            synthetic_lines = f.readlines()
        random.shuffle(synthetic_lines)
        train_split = int(0.8 * len(synthetic_lines))
        val_split = int(0.1 * len(synthetic_lines))

        all_train_lines.extend(synthetic_lines[:train_split])
        all_val_lines.extend(synthetic_lines[train_split:train_split + val_split])
        all_test_lines.extend(synthetic_lines[train_split + val_split:])
        print(f"INFO: Added {len(synthetic_lines)} lines from synthetic Hindi data.")
    else:
        print(f"WARNING: Synthetic Hindi data not found at {synthetic_hi_path}. Skipping.")


    # 2. Process DocHieNet Annotations
    dochienet_base_dir = os.path.join(RAW_DATA_DIR, 'dochienet_dataset')
    print("\nINFO: Starting DocHieNet processing.")
    # UNCOMMENT THE LINE BELOW TO ENABLE DOC HIENET PROCESSING
    dochienet_lines = process_doc_hienet(dochienet_base_dir) # Pass the base dir
    all_train_lines.extend(dochienet_lines)
    print("INFO: DocHieNet processing complete.")


    # 3. Process HJDataset Annotations
    hjdataset_base_dir = os.path.join(RAW_DATA_DIR, 'HJDataset')
    print("\nINFO: Starting HJDataset processing (Conceptual - requires full JSON parsing).")
    hjdataset_lines = process_hj_dataset(hjdataset_base_dir) # Pass the base dir
    all_train_lines.extend(hjdataset_lines)
    print("INFO: HJDataset processing placeholder complete.")


    # 4. Process Custom/Sample PDFs (e.g., hackathon's sample.pdf with sample.json ground truth)
    print("\nINFO: Processing sample.pdf using its ground truth JSON (sample.json) with robust matching.")
    sample_pdf_path_abs = os.path.join(RAW_DATA_DIR, 'sample.pdf') 
    sample_json_path = os.path.join(RAW_DATA_DIR, 'sample.json') 
    
    if os.path.exists(sample_pdf_path_abs) and os.path.exists(sample_json_path):
        page_w, page_h = get_pdf_dimensions(sample_pdf_path_abs)
        raw_elements_sample = extract_page_elements(sample_pdf_path_abs)

        if raw_elements_sample:
            lines_of_words_sample = group_elements_into_lines(raw_elements_sample, page_w, page_h)
            extracted_blocks_sample = group_lines_into_blocks(lines_of_words_sample, page_w, page_h)
            final_blocks_with_features_sample = extract_features_from_blocks(extracted_blocks_sample, page_w, page_h)

            with open(sample_json_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            sample_lines_for_fasttext = []
            
            gt_outline = ground_truth.get('outline', [])
            
            # Keep track of which extracted blocks have been used to match a GT entry
            matched_parser_block_indices = set() 
            actual_matched_gt_entries_count = 0 # Initialize here to avoid UnboundLocalError

            # Load the PDF again to search for GT text bounding boxes
            sample_doc_for_gt_bbox = None
            try:
                sample_doc_for_gt_bbox = fitz.open(sample_pdf_path_abs)
            except Exception as e:
                print(f"ERROR: Could not open sample.pdf for GT bbox search: {e}. Cannot perform accurate matching.")
                # If PDF cannot be opened for search, accurate matching is impossible.
                # All GT entries will be unmatched, added as warnings.
                # Proceeding with current values for matched_parser_block_indices/actual_matched_gt_entries_count

            # First pass: Match GT entries to extracted blocks using PyMuPDF's search_for and IoU
            for gt_entry in gt_outline:
                gt_text_orig = gt_entry['text'] # Original text from JSON
                gt_text_norm = gt_text_orig.replace('\n', ' ').strip()
                gt_level = gt_entry['level']
                gt_page_num_0_idx = gt_entry['page'] - 1 # Convert to 0-indexed page num
                
                # Skip H4 entries for now, as our model only predicts up to H3
                if gt_level == "H4":
                    continue 

                gt_bbox = None
                if sample_doc_for_gt_bbox:
                    gt_page_fitz = sample_doc_for_gt_bbox.load_page(gt_page_num_0_idx)
                    search_results = gt_page_fitz.search_for(gt_text_orig) 
                    if search_results:
                        gt_bbox = list(search_results[0]) # Use the first result's bbox
                
                best_match_block_idx = -1
                max_iou_score = 0.0
                max_text_similarity = 0.0 # Track best text similarity for debug output

                for i, parsed_block in enumerate(final_blocks_with_features_sample):
                    if i in matched_parser_block_indices: # Skip blocks already used
                        continue
                    
                    if parsed_block['page_num'] == gt_page_num_0_idx:
                        current_iou = 0.0
                        if gt_bbox:
                            current_iou = calculate_iou(gt_bbox, parsed_block['bbox'])

                        current_text_similarity = fuzz.token_sort_ratio(gt_text_norm.lower(), parsed_block['text'].lower()) / 100.0

                        # COMBINED MATCHING LOGIC (Tunable thresholds)
                        IOU_THRESHOLD = 0.5 # Minimum IoU for a good spatial match
                        TEXT_SIM_THRESHOLD = 0.7 # Minimum token sort ratio for good text match

                        # Prioritize: If strong IoU AND decent text match
                        if current_iou >= IOU_THRESHOLD and current_text_similarity >= TEXT_SIM_THRESHOLD:
                            if current_iou > max_iou_score: # Prioritize higher IoU for "best" match
                                max_iou_score = current_iou
                                max_text_similarity = current_text_similarity
                                best_match_block_idx = i
                        # Fallback: If no strong IoU but almost perfect text match (e.g., text fragmented but content same)
                        elif current_text_similarity >= 0.90 and current_iou < IOU_THRESHOLD: # Very high text sim (90%+)
                            if current_text_similarity > max_text_similarity: # Still prioritize best text sim
                                max_iou_score = current_iou # Record IoU for debug
                                max_text_similarity = current_text_similarity
                                best_match_block_idx = i

                # If a sufficiently good match is found
                if best_match_block_idx != -1 and (max_iou_score >= IOU_THRESHOLD or max_text_similarity >= 0.90): # Final confirmation
                    matched_parser_block_indices.add(best_match_block_idx)
                    best_match_block = final_blocks_with_features_sample[best_match_block_idx]
                    
                    fasttext_input = encode_features_for_fasttext(best_match_block['text'], best_match_block)
                    sample_lines_for_fasttext.append(f"__label__{gt_level} {fasttext_input}")
                    actual_matched_gt_entries_count += 1
                else:
                    print(f"WARNING: Sample PDF: Could not find a good match for GT entry: '{gt_text_norm}' ({gt_level}) on page {gt_page_num_0_idx} (Best IoU: {max_iou_score:.2f}, Best Text Sim: {max_text_similarity:.2f}).")
            
            if sample_doc_for_gt_bbox:
                sample_doc_for_gt_bbox.close() # Close the PDF document after searching

            # Second pass: Label all remaining unmatched blocks as BODY_TEXT
            body_text_count = 0
            for i, block in enumerate(final_blocks_with_features_sample):
                if i not in matched_parser_block_indices:
                    # Exclude very short blocks or blocks that are just page numbers
                    # A more robust check might involve font size, but for now, text length.
                    if len(block['text'].strip()) > 5 and not re.match(r'^\s*\d+[\.-]?\s*$', block['text'].strip()): # Filter out short numeric strings
                        fasttext_input = encode_features_for_fasttext(block['text'], block)
                        sample_lines_for_fasttext.append(f"__label__BODY_TEXT {fasttext_input}")
                        body_text_count += 1
            
            all_train_lines.extend(sample_lines_for_fasttext)
            print(f"INFO: Added {len(sample_lines_for_fasttext)} lines from sample.pdf (matched {actual_matched_gt_entries_count} GT entries, added {body_text_count} body text blocks).")
        else:
            print("WARNING: No elements extracted from sample.pdf by pdf_parser. Cannot add to training data.")
    else:
        print(f"WARNING: sample.pdf ({sample_pdf_path_abs}) or sample.json ({sample_json_path}) not found. Skipping sample PDF processing.")

# ... (rest of prepare_all_data and if __name__ == "__main__": block) ...


    # 5. Final Shuffle and Write to Files
    random.shuffle(all_train_lines)
    random.shuffle(all_val_lines)
    random.shuffle(all_test_lines)

    # Ensure processed directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        for line in all_train_lines:
            f.write(line.strip() + "\n")
    print(f"\nSUCCESS: Training data written to {TRAIN_FILE} ({len(all_train_lines)} lines).")

    with open(VAL_FILE, 'w', encoding='utf-8') as f:
        for line in all_val_lines:
            f.write(line.strip() + "\n")
    print(f"SUCCESS: Validation data written to {VAL_FILE} ({len(all_val_lines)} lines).")

    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        for line in all_test_lines:
            f.write(line.strip() + "\n")
    print(f"SUCCESS: Test data written to {TEST_FILE} ({len(all_test_lines)} lines).")


if __name__ == "__main__":
    # Ensure raw data dirs are created for convenience if manual download not done
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(RAW_DATA_DIR, 'dochienet_dataset', 'labels'), exist_ok=True) # Example subdirs
    os.makedirs(os.path.join(RAW_DATA_DIR, 'HJDataset', 'annotations'), exist_ok=True) # Example subdirs
    os.makedirs(os.path.join(SYNTHETIC_DATA_DIR), exist_ok=True) # Ensure synthetic dir is created if not by generator

    # Before running data_processor, ensure:
    # 1. synthetic_data_generator.py has been run and its output (synthetic_hi.txt) is in data/synthetic.
    # 2. DocHieNet and HJDataset are unzipped into data/raw.
    # 3. sample.pdf and sample.json (your E0H1CM114.json) are placed in data/raw/
    
    prepare_all_data()