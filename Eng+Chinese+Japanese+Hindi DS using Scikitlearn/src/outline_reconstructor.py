# src/outline_reconstructor.py (COMPLETE and CORRECT Code with relaxed filters for output)

import os
import sys
import json
import joblib 
import numpy as np 
from typing import List, Dict, Any 

# --- Fix for ModuleNotFoundError: Add project root to sys.path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
sys.path.insert(0, project_root)
# --- End of ModuleNotFoundError Fix ---

from src.pdf_parser import extract_page_elements, get_pdf_dimensions
from src.feature_extractor import group_elements_into_lines, group_lines_into_blocks, extract_features_from_blocks, encode_features_for_fasttext, THRESHOLD_INDENT # Import THRESHOLD_INDENT

# --- Configuration ---
MODEL_DIR = os.path.join(project_root, 'models')
PLACEHOLDER_MODEL_PATH = os.path.join(MODEL_DIR, 'heading_classifier_placeholder.pkl')

# Define heading levels with their hierarchy
HEADING_LEVELS = {
    "TITLE": 0,
    "H1": 1,
    "H2": 2,
    "H3": 3,
    "BODY_TEXT": 99 # A high number to indicate non-heading
}

def reconstruct_outline(pdf_path: str, model_path: str) -> Dict[str, Any]:
    """
    Processes a PDF, predicts heading levels, and reconstructs a hierarchical outline.
    Relaxed filters to ensure some output.
    """
    print(f"INFO: Starting outline reconstruction for {pdf_path}")

    # Load the trained model
    if not os.path.exists(model_path):
        print(f"ERROR: Model '{model_path}' not found. Please train it first.")
        return {"title": "N/A", "outline": [], "error": "Model not found"}
    
    try:
        classifier_model = joblib.load(model_path)
    except Exception as e:
        print(f"ERROR: Failed to load model from {model_path}: {e}")
        return {"title": "N/A", "outline": [], "error": f"Failed to load model: {e}"}

    # 1. Extract raw elements (words) from PDF
    page_w, page_h = get_pdf_dimensions(pdf_path) 
    raw_elements = extract_page_elements(pdf_path)

    if not raw_elements:
        print("WARNING: No raw elements extracted. Cannot reconstruct outline.")
        return {"title": "N/A", "outline": [], "error": "No text extracted from PDF"}

    # 2. Group words into lines and then blocks
    lines_of_words = group_elements_into_lines(raw_elements, page_w, page_h)
    extracted_blocks = group_lines_into_blocks(lines_of_words, page_w, page_h)
    
    if not extracted_blocks:
        print("WARNING: No text blocks formed. Cannot reconstruct outline.")
        return {"title": "N/A", "outline": [], "error": "No text blocks formed"}

    # 3. Extract advanced features for each block
    final_blocks_with_features = extract_features_from_blocks(extracted_blocks, page_w, page_h)

    # 4. Predict heading levels
    texts_to_predict = []
    original_blocks_map = [] 
    for block in final_blocks_with_features:
        encoded_text = encode_features_for_fasttext(block['text'], block)
        texts_to_predict.append(encoded_text)
        original_blocks_map.append(block)
    
    if not texts_to_predict:
        print("WARNING: No text to predict. Cannot reconstruct outline.")
        return {"title": "N/A", "outline": [], "error": "No text to predict"}

    predicted_labels = classifier_model.predict(texts_to_predict)
    predicted_probs = classifier_model.predict_proba(texts_to_predict) # Get probabilities

    # Structure predicted blocks for outline generation
    predicted_blocks_with_labels = []
    for i, block in enumerate(original_blocks_map):
        label = predicted_labels[i]
        prob = np.max(predicted_probs[i]) if predicted_probs.size > 0 else 0.0
        
        predicted_blocks_with_labels.append({
            'text': block['text'],
            'bbox': block['bbox'],
            'page_num': block['page_num'],
            'font_size': block['font_size'],
            'x_indentation': block['x_indentation'],
            'y_spacing_relative': block['y_spacing_relative'],
            'is_list_item': block.get('is_list_item', False), 
            'is_all_caps': block.get('is_all_caps', False), 
            'is_centered': block.get('is_centered', False), 
            'predicted_label': label,
            'predicted_prob': prob,
            'level_rank': HEADING_LEVELS.get(label, 99) 
        })
    
    # Sort blocks primarily by page_num, then y-coordinate, then x-coordinate
    predicted_blocks_with_labels.sort(key=lambda b: (b['page_num'], b['bbox'][1], b['bbox'][0]))


    # 5. Reconstruct hierarchical outline (The "Rule-Based" post-processing)
    outline = []
    document_title = "N/A"
    
    # --- Refined Title Extraction ---
    # Find potential titles from predicted blocks on page 0, highest font size, top
    title_candidates = sorted([
        b for b in predicted_blocks_with_labels 
        if b['page_num'] == 0 and b['bbox'][1] < page_h / 3 and b['predicted_label'] in ["TITLE", "H1"] # Top third of the first page
    ], key=lambda b: (-b['font_size'], b['bbox'][1])) # Sort by largest font first, then top-to-bottom

    if title_candidates:
        # Prioritize 'TITLE' prediction with reasonable confidence, then 'H1'
        for candidate in title_candidates:
            if candidate['predicted_label'] == "TITLE" and candidate['predicted_prob'] > 0.5: # Relaxed confidence for Title
                document_title = candidate['text']
                candidate['level_rank'] = -1 # Mark as used for title
                break
            elif candidate['predicted_label'] == "H1" and candidate['predicted_prob'] > 0.7 and document_title == "N/A": # Relaxed H1 confidence
                document_title = candidate['text']
                candidate['level_rank'] = -1 # Mark as used for title
                break
        
        # Fallback if still no title: Take the highest font size block from top candidates
        if document_title == "N/A" and title_candidates:
            document_title = title_candidates[0]['text']
            title_candidates[0]['level_rank'] = -1


    # --- Filter and enforce hierarchical structure for outline entries ---
    final_outline_entries_raw = [] # Using the correct variable name

    # Track the last valid heading level added to enforce hierarchy (stack-like behavior)
    current_active_headings_stack = [{"level": "ROOT", "level_rank": HEADING_LEVELS["TITLE"], "page": -1, "y_pos": -1}] 
    
    for block_data in predicted_blocks_with_labels:
        # Skip blocks used as document title
        if block_data['level_rank'] == -1: 
            continue

        label = block_data['predicted_label']
        text = block_data['text']
        page = block_data['page_num'] + 1 
        current_level_rank = HEADING_LEVELS.get(label, 99)

        # HEURISTIC 1: Filter out very long "headings" (likely misclassified body text)
        # RELAXED FILTER: Max 30 words for a heading, or very high indentation for main heading.
        if label != "BODY_TEXT" and \
           (len(text.split()) > 30 or block_data['x_indentation'] > THRESHOLD_INDENT * 2.5): # Relaxed thresholds
            # print(f"DEBUG: Filtering long/indented misclassified heading: '{text}' ({label})")
            continue 

        # HEURISTIC 2: Filter out headings with very low prediction probability
        # RELAXED CONFIDENCE: Allow lower confidence to see more output.
        if label != "BODY_TEXT" and block_data['predicted_prob'] < 0.5: # Lower confidence threshold
             # print(f"DEBUG: Filtering low-confidence heading: '{text}' ({label}, Prob: {block_data['predicted_prob']:.2f})")
             continue

        # HEURISTIC 3: Filter out list items misclassified as headings (if is_list_item is reliable)
        # This assumes is_list_item is correctly populated by feature_extractor.py
        if label != "BODY_TEXT" and block_data.get('is_list_item', False) and block_data['predicted_prob'] < 0.7: # Only filter list items if confidence is low
            # print(f"DEBUG: Filtering list item misclassified as heading: '{text}' ({label})")
            continue

        # HEURISTIC 4: Enforce Hierarchy using a stack-based approach
        if label != "BODY_TEXT" and current_level_rank < HEADING_LEVELS["BODY_TEXT"]: # It's a predicted heading
            
            # Adjust stack: Pop headings from stack that are of higher (lower number) or equal rank 
            # if the current heading is at a higher level (lower rank) or if there's a large jump in vertical position
            while len(current_active_headings_stack) > 1 and \
                  (current_level_rank < current_active_headings_stack[-1]['level_rank'] or # Current is higher level (e.g. H1 after H2)
                   (current_level_rank == current_active_headings_stack[-1]['level_rank'] and block_data['page_num'] > current_active_headings_stack[-1]['page']) or # Same level but on new page
                   (current_level_rank == current_active_headings_stack[-1]['level_rank'] and block_data['page_num'] == current_active_headings_stack[-1]['page'] and block_data['y_spacing_relative'] > 0.05) # Same level, same page, large gap
                  ):
                current_active_headings_stack.pop()
            
            # If current heading is at a significantly lower level (e.g., H1 -> H3 directly without H2)
            # This helps to prevent false positives that skip levels
            if current_level_rank > current_active_headings_stack[-1]['level_rank'] + 1 : # e.g., H1 -> H3 (skips H2)
                # print(f"DEBUG: Skipping '{text}' ({label}) due to large jump in hierarchy from {current_active_headings_stack[-1]['level']} to {label}.")
                pass # RELAXED: Don't skip, just accept and let hierarchy reset

            # Add the current heading to the outline and update stack
            final_outline_entries_raw.append({
                "level": label,
                "text": text,
                "page": page,
                "bbox": block_data['bbox'] # Include bbox for robust deduplication
            })
            
            # Push to stack
            current_active_headings_stack.append({"level": label, "level_rank": current_level_rank, "page": block_data['page_num'], "y_pos": block_data['bbox'][1]})
            current_active_headings_stack = [current_active_headings_stack[0]] + sorted(current_active_headings_stack[1:], key=lambda x: x['level_rank']) # Keep sorted by rank

    # Deduplicate entries based on text and page to clean up final output.
    # The malformed JSON was due to a `sys.path` typo and aggressive filtering previously.
    deduplicated_outline = []
    seen_entries = set() # Store (normalized_text, page_num, approx_y_pos) tuples for robust deduplication

    for entry in final_outline_entries_raw: # Correct variable name here
        # Create a more robust key for deduplication
        # Use a rounded y-coordinate or a hash of the bbox for approximate position on page
        approx_y_pos = int(entry['bbox'][1] / page_h * 1000) # Scale to 0-1000 range for page
        normalized_text_page_pos = (entry['text'].strip().lower(), entry['page'], approx_y_pos)

        if normalized_text_page_pos not in seen_entries:
            deduplicated_outline.append(entry)
            seen_entries.add(normalized_text_page_pos)

    return {"title": document_title, "outline": deduplicated_outline}

# --- Example Usage ---
if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir)) 
    sample_pdf_path = os.path.join(project_root, 'data', 'raw', 'sample.pdf')

    if not os.path.exists(PLACEHOLDER_MODEL_PATH):
        print(f"ERROR: Model '{PLACEHOLDER_MODEL_PATH}' not found. Please run src/data_processor.py and src/heading_classifier.py first.")
    else:
        output_json = reconstruct_outline(sample_pdf_path, PLACEHOLDER_MODEL_PATH)
        
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, 'sample_outline.json')
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=4)
        
        print(f"\nSUCCESS: Outline reconstructed and saved to {output_file_path}")
        print("--- Generated Outline ---")
        print(json.dumps(output_json, indent=2, ensure_ascii=False))