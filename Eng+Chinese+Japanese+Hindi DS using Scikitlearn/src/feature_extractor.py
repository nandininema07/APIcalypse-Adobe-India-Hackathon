# src/feature_extractor.py (COMPLETE and CORRECT Code with refined block grouping)

import os
import sys
import math
import re # Ensure re is imported for regex functions
from typing import List, Dict, Any

# --- Fix for ModuleNotFoundError: Add project root to sys.path ---
# Get the absolute path of the directory containing the current script (src/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (e.g., 'Rule Based + ML Approach/')
project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir))
# Add the project root to sys.path
sys.path.insert(0, project_root)
# --- End of ModuleNotFoundError Fix ---

# Import get_pdf_dimensions and extract_page_elements from pdf_parser.py
from src.pdf_parser import get_pdf_dimensions, extract_page_elements

# Define thresholds (these might need fine-tuning based on dataset analysis)
THRESHOLD_LARGE_GAP = 0.05 # Relative vertical gap (e.g., new section, title)
THRESHOLD_MEDIUM_GAP = 0.02 # Relative vertical gap (e.g., new paragraph/heading)
THRESHOLD_SMALL_GAP = 0.005 # Relative minimal line spacing

THRESHOLD_INDENT = 0.02 # Relative significant horizontal indentation

# Functions to group words into lines and then lines into blocks
def group_elements_into_lines(elements: List[Dict[str, Any]], page_width: float, page_height: float, line_gap_threshold: float = 0.01) -> List[List[Dict[str, Any]]]:
    """
    Groups word-level elements into lines based on vertical proximity and page number.
    Assumes elements are already sorted by page_num, then y0, then x0.
    """
    lines: List[List[Dict[str, Any]]] = []
    current_line: List[Dict[str, Any]] = []

    if not elements:
        return []

    # Sort elements by page_num, then top-y (y0), then left-x (x0)
    elements.sort(key=lambda e: (e['page_num'], e['bbox'][1], e['bbox'][0]))

    for i, element in enumerate(elements):
        if not current_line:
            current_line.append(element)
        else:
            prev_element = current_line[-1]
            # Check if element is on the same page and vertically close enough to be on the same line
            # And if it's horizontally aligned (within some tolerance, or just next to previous)
            
            # Line grouping tolerance should be relative to font size/line height, not page height
            # For words, checking if their y-coordinates overlap or are very close
            
            # If vertical centers are close, or y-overlap is significant
            y_center_diff = abs((element['bbox'][1] + element['bbox'][3])/2 - (prev_element['bbox'][1] + prev_element['bbox'][3])/2)
            y_overlap = min(element['bbox'][3], prev_element['bbox'][3]) - max(element['bbox'][1], prev_element['bbox'][1])
            
            # Consider elements on same line if: same page AND (centers are close OR they vertically overlap)
            if element['page_num'] == prev_element['page_num'] and \
               (y_center_diff < line_gap_threshold * page_height or y_overlap > 0): # line_gap_threshold used as a flexible 'closeness' factor
                current_line.append(element)
            else:
                lines.append(sorted(current_line, key=lambda e: e['bbox'][0])) # Sort words in line by x0
                current_line = [element]
    if current_line:
        lines.append(sorted(current_line, key=lambda e: e['bbox'][0]))
    return lines


def group_lines_into_blocks(lines: List[List[Dict[str, Any]]], page_width: float, page_height: float, 
                             block_gap_threshold: float = 0.02, # Relative threshold for large vertical gaps
                             indentation_change_threshold: float = 0.01, # Relative change in x0 to trigger new block
                             short_line_max_width_ratio: float = 0.5, # Max width ratio for a "short line" (e.g., heading)
                             max_lines_in_block: int = 15 # Max lines in a block to prevent huge blocks
                            ) -> List[Dict[str, Any]]:
    """
    Groups lines into logical blocks based on vertical spacing, page changes, and layout heuristics.
    More aggressively separates blocks based on gaps, indentation, and line length.
    """
    blocks: List[Dict[str, Any]] = []
    current_block_lines: List[List[Dict[str, Any]]] = []

    if not lines:
        return []
    
    # Pre-calculate average line height for adaptive thresholding
    all_line_heights = []
    for line in lines:
        if line:
            line_top = min(e['bbox'][1] for e in line)
            line_bottom = max(e['bbox'][3] for e in line)
            if line_bottom > line_top:
                all_line_heights.append(line_bottom - line_top)
    avg_line_height = sum(all_line_heights) / len(all_line_heights) if all_line_heights else (12 / 72 * page_height) # Default if no lines

    for i, line in enumerate(lines):
        if not line:
            continue

        line_y0 = min(e['bbox'][1] for e in line)
        line_y1 = max(e['bbox'][3] for e in line)
        line_x0 = min(e['bbox'][0] for e in line)
        line_x1 = max(e['bbox'][2] for e in line)
        line_page_num = line[0]['page_num']
        line_text = " ".join(e['text'] for e in line).strip() 
        
        current_line_indent_ratio = line_x0 / page_width
        current_line_width_ratio = (line_x1 - line_x0) / page_width
        is_current_line_short = current_line_width_ratio < short_line_max_width_ratio


        start_new_block = False

        if not current_block_lines:
            # Always start a new block for the very first line
            start_new_block = True
        else:
            prev_line = current_block_lines[-1]
            prev_line_y1 = max(e['bbox'][3] for e in prev_line)
            prev_line_x0 = min(e['bbox'][0] for e in prev_line)
            prev_line_page_num = prev_line[0]['page_num']
            # prev_line_text = " ".join(e['text'] for e in prev_line).strip() # Not strictly needed for logic here
            
            prev_line_width_ratio = (max(e['bbox'][2] for e in prev_line) - min(e['bbox'][0] for e in prev_line)) / page_width
            is_prev_line_short = prev_line_width_ratio < short_line_max_width_ratio


            vertical_gap = line_y0 - prev_line_y1
            # Relative vertical gap based on page height AND average line height
            relative_vertical_gap_page = vertical_gap / page_height
            relative_vertical_gap_line = vertical_gap / avg_line_height if avg_line_height > 0 else 0


            # Conditions to start a new block:
            # 1. New page
            if line_page_num != prev_line_page_num:
                start_new_block = True
            # 2. Significant vertical gap (more than threshold relative to page height OR more than X times avg line height)
            elif relative_vertical_gap_page > block_gap_threshold or relative_vertical_gap_line > 2.0: # 2x average line height is a strong break
                start_new_block = True
            # 3. Indentation shift: Current line shifts significantly left compared to previous.
            #    This often indicates a new heading or paragraph.
            elif current_line_indent_ratio < (prev_line_x0 / page_width) - indentation_change_threshold:
                start_new_block = True
            # 4. Current block is getting too long (to prevent combining huge paragraphs)
            elif len(current_block_lines) >= max_lines_in_block:
                start_new_block = True
            # 5. Heuristic: If previous line was a short line (potential heading) and current line is NOT short or has significant gap
            #    This helps to properly break after a single-line heading.
            elif is_prev_line_short and not is_current_line_short and relative_vertical_gap_line > 0.5: # Half line height gap after short line
                start_new_block = True
            # 6. Heuristic: Current line is very short and significantly indented compared to previous.
            #    Often signals a list item or sub-point if prev wasn't.
            elif is_current_line_short and current_line_indent_ratio > (prev_line_x0 / page_width) + indentation_change_threshold:
                start_new_block = True


        if start_new_block:
            # Finalize previous block if it has content
            block_text_lines = [" ".join(e['text'] for e in l).strip() for l in current_block_lines if l]
            block_text = " ".join(block_text_lines).strip()
            
            if block_text: # Only add non-empty blocks
                block_bbox = [
                    min(e['bbox'][0] for l in current_block_lines for e in l),
                    min(e['bbox'][1] for l in current_block_lines for e in l),
                    max(e['bbox'][2] for l in current_block_lines for e in l),
                    max(e['bbox'][3] for l in current_block_lines for e in l)
                ]
                font_sizes_in_block = [e['font_size'] for l in current_block_lines for e in l if e['font_size'] > 0]
                dominant_font_size = max(font_sizes_in_block) if font_sizes_in_block else 0.0

                blocks.append({
                    'text': block_text,
                    'bbox': block_bbox,
                    'page_num': prev_line_page_num if current_block_lines else line_page_num, # Page of the last line of the previous block
                    'font_size': dominant_font_size,
                    'is_bold': False, # Cannot infer directly from words
                    'is_italic': False, # Cannot infer directly from words
                })
            current_block_lines = [line] # Start new block with current line
        else:
            current_block_lines.append(line)

    # Add the last block (if any lines remain)
    if current_block_lines:
        block_text_lines = [" ".join(e['text'] for e in l).strip() for l in current_block_lines if l]
        block_text = " ".join(block_text_lines).strip()
        if block_text: # Only add if not empty
            block_bbox = [
                min(e['bbox'][0] for l in current_block_lines for e in l),
                min(e['bbox'][1] for l in current_block_lines for e in l),
                max(e['bbox'][2] for l in current_block_lines for e in l),
                max(e['bbox'][3] for l in current_block_lines for e in l)
            ]
            font_sizes_in_block = [e['font_size'] for l in current_block_lines for e in l if e['font_size'] > 0]
            dominant_font_size = max(font_sizes_in_block) if font_sizes_in_block else 0.0

            blocks.append({
                'text': block_text,
                'bbox': block_bbox,
                'page_num': line_page_num, # Page num of the last line added to this block
                'font_size': dominant_font_size,
                'is_bold': False,
                'is_italic': False,
            })
    return blocks

# ... (rest of feature_extractor.py, including extract_features_from_blocks and encode_features_for_fasttext, and example usage) ...


def extract_features_from_blocks(blocks: List[Dict[str, Any]], page_width: float, page_height: float) -> List[Dict[str, Any]]:
    """
    Calculates advanced features for each block and encodes them.
    """
    processed_blocks = []
    
    # Calculate global max font size for relative sizing
    all_font_sizes = [b['font_size'] for b in blocks if b['font_size'] > 0]
    max_doc_font_size = max(all_font_sizes) if all_font_sizes else 1.0 # Avoid division by zero

    for i, block in enumerate(blocks):
        # Calculate relative font size
        block['relative_font_size'] = block['font_size'] / max_doc_font_size if max_doc_font_size > 0 else 0.0

        # Check for all caps
        block['is_all_caps'] = block['text'].isupper() and len(block['text']) > 3 # Avoid short acronyms

        # Check for numbering/bullet points (basic regex)
        starts_with_list_marker = bool(
            block['text'] and (
                block['text'].lstrip().startswith(('•', '-', '*', '–', '—')) or # Bullet points
                re.match(r'^\s*(\d+\.?\d*|\([a-zA-Z]\))\s+', block['text']) # Numbering like "1.", "1.1", "(a)"
            )
        )
        block['is_list_item'] = starts_with_list_marker

        # Calculate x_indentation (relative to page width)
        block['x_indentation'] = block['bbox'][0] / page_width 

        # Calculate y_spacing_relative (relative to page height)
        # For the very first block of the document (or first block on a new page),
        # this is the distance from the top of the content area.
        # For subsequent blocks, it's the gap to the previous block.
        if i == 0 or block['page_num'] != blocks[i-1]['page_num']: # First block or new page
            block['y_spacing_relative'] = block['bbox'][1] / page_height # Distance from top of page
        else:
            prev_block = blocks[i-1]
            y_spacing = block['bbox'][1] - prev_block['bbox'][3]
            block['y_spacing_relative'] = max(0.0, y_spacing / page_height) # Ensure no negative spacing
        
        # Check if centered (within a tolerance)
        block_center_x = (block['bbox'][0] + block['bbox'][2]) / 2
        page_center_x = page_width / 2
        # A 5% tolerance is arbitrary, refine if needed.
        block['is_centered'] = abs(block_center_x - page_center_x) < (page_width * 0.05) 

        processed_blocks.append(block)
    return processed_blocks


def encode_features_for_fasttext(text_content: str, features: Dict[str, Any]) -> str:
    """
    Encodes layout features as special tokens prepended to the text content.
    """
    encoded_tokens = []

    # Bin Font Size (using inferred font_size from words mode)
    # The exact thresholds might need tuning based on actual data distribution.
    if features.get("font_size", 0.0) >= 20.0:
        encoded_tokens.append("<FS_LARGE>")
    elif features.get("font_size", 0.0) >= 14.0:
        encoded_tokens.append("<FS_MEDIUM>")
    elif features.get("font_size", 0.0) >= 10.0:
        encoded_tokens.append("<FS_SMALL>")
    else:
        encoded_tokens.append("<FS_XSMALL>")

    # Boolean flags (will likely be False as derived from "words" mode)
    if features.get("is_bold", False):
        encoded_tokens.append("<BOLD>")
    if features.get("is_italic", False):
        encoded_tokens.append("<ITALIC>")
    if features.get("is_centered", False):
        encoded_tokens.append("<CENTERED>")
    if features.get("is_list_item", False):
        encoded_tokens.append("<BULLET>")
    if features.get("is_all_caps", False):
        encoded_tokens.append("<ALLCAPS>")

    # Y-Spacing (binned based on relative gap)
    if features.get('y_spacing_relative', 0.0) >= THRESHOLD_LARGE_GAP:
        encoded_tokens.append("<LGAP>")
    elif features.get('y_spacing_relative', 0.0) >= THRESHOLD_MEDIUM_GAP:
        encoded_tokens.append("<MGAP>")
    elif features.get('y_spacing_relative', 0.0) >= THRESHOLD_SMALL_GAP:
        encoded_tokens.append("<SGAP>")
    else:
        encoded_tokens.append("<TGAP>") # Tiny gap or no significant gap

    # X-Indentation (binned based on relative indentation)
    if features.get('x_indentation', 0.0) >= THRESHOLD_INDENT:
        encoded_tokens.append("<INDENT>")
    else:
        encoded_tokens.append("<NINDENT>") # No significant indentation

    # Clean text content to remove extra spaces/newlines that might mess up FastText
    cleaned_text = " ".join(text_content.split()).replace('\n', ' ').replace('\r', '')

    return " ".join(encoded_tokens + [cleaned_text])


# --- Example Usage (for testing feature_extractor.py) ---
if __name__ == "__main__":
    # Ensure sys.path is set correctly for imports in standalone run
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir)) 
    sys.path.insert(0, project_root)
    
    # Import extract_page_elements AFTER sys.path is set
    from src.pdf_parser import extract_page_elements 

    sample_pdf_path = os.path.join(project_root, 'data', 'raw', 'sample.pdf')
    
    # Get page dimensions from the PDF using the function from pdf_parser
    page_w, page_h = get_pdf_dimensions(sample_pdf_path)

    print(f"DEBUG: Using page dimensions W:{page_w}, H:{page_h} for feature calculations.")

    # 1. Extract raw elements using pdf_parser
    raw_elements = extract_page_elements(sample_pdf_path)

    if not raw_elements:
        print("ERROR: No raw elements extracted by pdf_parser. Cannot proceed with feature extraction.")
    else:
        print(f"\nDEBUG: Successfully extracted {len(raw_elements)} raw elements. Grouping into lines...")
        # 2. Group raw words into lines
        lines_of_words = group_elements_into_lines(raw_elements, page_w, page_h)
        print(f"DEBUG: Grouped into {len(lines_of_words)} lines. Grouping into blocks...")

        # 3. Group lines into blocks
        extracted_blocks = group_lines_into_blocks(lines_of_words, page_w, page_h)
        print(f"DEBUG: Grouped into {len(extracted_blocks)} blocks. Extracting advanced features...")

        # 4. Extract advanced features for each block
        final_blocks_with_features = extract_features_from_blocks(extracted_blocks, page_w, page_h)
        print(f"DEBUG: Extracted advanced features for {len(final_blocks_with_features)} blocks.")

        print("\n--- Processed Blocks with Encoded Features (First 10) ---")
        for i, block in enumerate(final_blocks_with_features[:10]):
            # Assign a dummy label for demonstration (in real data, this comes from annotations)
            # This is a very simple heuristic just for this example output
            dummy_label = "BODY_TEXT" 
            if i == 0 and block['font_size'] >= 24: dummy_label = "TITLE"
            elif block['font_size'] >= 18 and block['x_indentation'] < 0.1: dummy_label = "H1"
            elif block['font_size'] >= 14 and block['x_indentation'] < 0.15: dummy_label = "H2"
            elif block['font_size'] >= 12 and block['x_indentation'] < 0.2: dummy_label = "H3"


            encoded_string = encode_features_for_fasttext(block['text'], block)
            print(f"--- Block {i+1} (Page {block['page_num']}) ---")
            print(f"  Approx Font Size: {block['font_size']:.2f}")
            print(f"  Relative Y-Spacing: {block['y_spacing_relative']:.4f}")
            print(f"  X-Indentation: {block['x_indentation']:.4f}")
            print(f"  Is All Caps: {block['is_all_caps']}")
            print(f"  Is Centered: {block['is_centered']}")
            print(f"  Is List Item: {block['is_list_item']}")
            print(f"  Raw Text: '{block['text']}'")
            print(f"  FastText Input: __label__{dummy_label} {encoded_string}")
            print("-" * 30)

        if len(final_blocks_with_features) > 10:
            print("... (showing only first 10 blocks)")