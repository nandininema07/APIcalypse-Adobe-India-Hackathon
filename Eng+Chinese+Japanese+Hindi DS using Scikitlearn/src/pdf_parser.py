# src/pdf_parser.py (COMPLETE and CORRECT Code with OCR Fallback and Optimization)

import fitz # PyMuPDF
from typing import List, Dict, Any
import os
import pytesseract # ADDED for OCR
from PIL import Image # ADDED for image processing with OCR
import numpy as np # ADDED for image array operations

# --- Helper Function: Get Page Dimensions ---
def get_pdf_dimensions(pdf_path: str):
    """Safely gets page dimensions (width, height) from a PDF's first page."""
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count > 0:
            page = doc.load_page(0)
            dims = page.rect # Returns (x0, y0, x1, y1)
            doc.close()
            return dims.width, dims.height
        doc.close()
        return 595.0, 842.0 # Default A4 if no pages
    except Exception as e:
        # print(f"WARNING: Could not get actual page dimensions for {pdf_path}: {e}. Using default A4.") # Commented for cleaner output
        return 595.0, 842.0 # Default A4 dimensions (approx 8.27 x 11.69 inches at 72dpi)

# --- OCR Function (NEW) ---
def ocr_page_for_words(page: fitz.Page, lang: str = 'eng') -> List[Dict[str, Any]]:
    """
    Performs OCR on a PDF page's image and returns extracted words with bounding boxes.
    Args:
        page (fitz.Page): The PyMuPDF page object.
        lang (str): Tesseract language code (e.g., 'eng', 'jpn', 'hin').
    Returns:
        List[Dict[str, Any]]: List of extracted words with 'text', 'bbox', etc.
    """
    print(f"DEBUG_OCR: Attempting OCR on page {page.number} with lang='{lang}'...")
    
    # --- OPTIMIZATION: Reduced resolution for faster OCR ---
    # From (2,2) to (1.5, 1.5) or (1,1). Test (1.5, 1.5) first.
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)) 
    
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    img_pil = Image.fromarray(img_array)

    try:
        data = pytesseract.image_to_data(img_pil, lang=lang, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        print("ERROR_OCR: Tesseract is not installed or not in your PATH. Please install it. Skipping OCR.")
        return []
    except Exception as e:
        print(f"ERROR_OCR: Tesseract OCR failed on page {page.number}: {e}. Skipping OCR.")
        return []

    elements = []
    # Loop through extracted text data
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        # Level 5 is typically word level
        if data['level'][i] == 5 and data['text'][i].strip():
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            
            # Convert OCR pixel coordinates back to PDF points (PyMuPDF's coordinate system)
            # The scaling factor is `page.rect.width / pix.width` (for x) and `page.rect.height / pix.h` (for y)
            # which accounts for the resolution scaling in get_pixmap
            scale_x_to_pdf_points = page.rect.width / pix.width
            scale_y_to_pdf_points = page.rect.height / pix.h

            bbox_pdf = [
                x * scale_x_to_pdf_points,
                y * scale_y_to_pdf_points,
                (x + w) * scale_x_to_pdf_points,
                (y + h) * scale_y_to_pdf_points
            ]

            elements.append({
                'text': data['text'][i].strip(),
                'bbox': bbox_pdf,
                'page_num': page.number,
                'font_name': 'OCR_Inferred', # Indicate source
                'font_size': h * scale_y_to_pdf_points, # Approximate font size from pixel height
                'is_bold': False, # OCR typically doesn't give bold/italic flags
                'is_italic': False,
                'span_id': f"ocr_p{page.number}_w{i}",
                'block_id': data['block_num'][i],
                'line_id': data['line_num'][i],
            })
    print(f"DEBUG_OCR: OCR extracted {len(elements)} words from page {page.number}.")
    return elements


def extract_page_elements(pdf_path: str, lang: str = 'eng') -> List[Dict[str, Any]]:
    """
    Extracts detailed text elements and their layout metadata from a PDF.
    Attempts rawdict, then words, then OCR as fallbacks.
    """
    all_elements: List[Dict[str, Any]] = []

    print(f"DEBUG: Attempting to open PDF at: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        print(f"DEBUG: Successfully opened PDF. Page count: {doc.page_count}")
    except Exception as e:
        print(f"ERROR: Failed to open PDF {pdf_path}: {e}")
        return []

    if doc.page_count == 0:
        print(f"WARNING: PDF {pdf_path} has no pages.")
        doc.close()
        return []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_extracted_elements = []
        
        # --- Attempt 1: Rawdict (for rich features like bold/italic) ---
        raw_dict_had_valid_text = False
        try:
            raw_dict = page.get_text("rawdict")
            if raw_dict and 'blocks' in raw_dict:
                for block_idx, block in enumerate(raw_dict.get('blocks', [])):
                    if block.get('type') == 0: # Text block type
                        for line_idx, line in enumerate(block.get('lines', [])):
                            for span_idx, span in enumerate(line.get('spans', [])):
                                span_text_raw = span.get('text')
                                if span_text_raw and span_text_raw.strip(): # Check if text exists and is not empty/None
                                    text = span_text_raw.strip()
                                    raw_dict_had_valid_text = True
                                    span_flags = span.get('flags', 0)
                                    element = {
                                        'text': text, 'bbox': list(span.get('bbox', [0.0]*4)), 'page_num': page_num,
                                        'font_name': span.get('font', 'N/A_rawdict'), 'font_size': span.get('size', 0.0),
                                        'is_bold': bool(span_flags & fitz.TEXT_FONT_BOLD), 'is_italic': bool(span_flags & fitz.TEXT_FONT_ITALIC),
                                        'span_id': f"rawdict_b{block_idx}_l{line_idx}_s{span_idx}", 'block_id': block_idx, 'line_id': line_idx,
                                    }
                                    page_extracted_elements.append(element)
        except Exception as e:
            print(f"DEBUG: Error during rawdict processing on page {page_num}: {e}.")
            raw_dict_had_valid_text = False 

        if raw_dict_had_valid_text:
            all_elements.extend(page_extracted_elements)
            print(f"DEBUG: Page {page_num}: Extracted {len(page_extracted_elements)} elements using 'rawdict'.")
            continue # Move to next page if rawdict worked

        # --- Attempt 2: Words mode (more robust for text content/bbox) ---
        print(f"DEBUG: Page {page_num}: 'rawdict' yielded no valid text. Attempting fallback to 'words'.")
        words = page.get_text("words") # (x0, y0, x1, y1, word, block_no, line_no, word_no)
        
        if words:
            print(f"DEBUG: Page {page_num}: Fallback 'words' extraction found {len(words)} words.")
            for word_bbox_info in words:
                text = word_bbox_info[4].strip()
                if not text: continue
                bbox = list(word_bbox_info[:4])
                inferred_font_size = bbox[3] - bbox[1] if bbox[3] > bbox[1] else 0.0 
                element = {
                    'text': text, 'bbox': bbox, 'page_num': page_num,
                    'font_name': 'Inferred_from_words', 'font_size': inferred_font_size,
                    'is_bold': False, 'is_italic': False, # Not determinable from "words"
                    'span_id': f"words_p{page_num}_w{word_bbox_info[7]}", 'block_id': word_bbox_info[5], 'line_id': word_bbox_info[6],
                }
                page_extracted_elements.append(element)
            all_elements.extend(page_extracted_elements)
            continue # Move to next page if words mode worked

        # --- Attempt 3: OCR Fallback (for image-based/scanned pages) ---
        print(f"DEBUG: Page {page_num}: 'words' also yielded no text. Attempting OCR fallback.")
        ocr_elements = ocr_page_for_words(page, lang=lang) # Use specified language for OCR
        if ocr_elements:
            print(f"DEBUG: Page {page_num}: OCR extracted {len(ocr_elements)} words. Adding to elements.")
            all_elements.extend(ocr_elements)
        else:
            print(f"WARNING: Page {page_num}: OCR also yielded no text. This page might be entirely non-textual or problematic.")

    doc.close()
    return all_elements

# --- Example Usage (for testing pdf_parser.py) ---
if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, os.pardir)) 
    sample_pdf_path = os.path.join(project_root, 'data', 'raw', 'sample.pdf')

    print(f"DEBUG: Calculated absolute path for sample.pdf: {sample_pdf_path}")

    try:
        os.makedirs(os.path.dirname(sample_pdf_path), exist_ok=True)

        create_dummy_pdf = False 

        if os.path.exists(sample_pdf_path) and not create_dummy_pdf:
            print(f"Using existing PDF for testing: {sample_pdf_path}")
        else:
            print(f"WARNING: '{sample_pdf_path}' not found at calculated absolute path, OR 'create_dummy_pdf' is True. Fallback: Creating a very basic dummy PDF.")
            dummy_doc = fitz.open()
            page = dummy_doc.new_page()
            page.insert_text((50, 50), "Dummy Title", fontsize=24)
            page.insert_text((50, 100), "Dummy Section", fontsize=18)
            dummy_doc.save(sample_pdf_path)
            dummy_doc.close()
            print("Dummy PDF created/recreated (fallback). You can now run the parser.")

        # Test with English for sample.pdf
        elements = extract_page_elements(sample_pdf_path, lang='eng') 
        if elements:
            print(f"Extracted {len(elements)} elements from {sample_pdf_path}:")
            for i, elem in enumerate(elements[:20]):
                print(f"--- Element {i+1} ---")
                print(f"  Text: '{elem.get('text', 'KEY_MISSING_OR_NONE')}'")
                print(f"  Page: {elem.get('page_num', 'N/A')}")
                print(f"  BBox: {elem.get('bbox', 'N/A')}")
                print(f"  Font: {elem.get('font_name', 'N/A')} (Size: {elem.get('font_size', 'N/A')})")
                print(f"  Bold: {elem.get('is_bold', 'N/A')}, Italic: {elem.get('is_italic', 'N/A')}")
            if len(elements) > 20:
                print("... (showing only first 20 elements)")
        else:
            print("No elements extracted or error occurred. This might indicate an entirely image-based PDF where even 'words' fails.")

    except ImportError:
        print("Please install PyMuPDF, pytesseract, Pillow: pip install PyMuPDF pytesseract Pillow")
    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}")