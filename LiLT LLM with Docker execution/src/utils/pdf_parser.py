# src/utils/pdf_parser.py
import pypdfium2 as pdfium
import logging

logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self):
        pass

    def load_pdf(self, pdf_path):
        """Loads a PDF document."""
        try:
            doc = pdfium.PdfDocument(pdf_path)
            logger.info(f"Loaded PDF: {pdf_path} with {len(doc)} pages.")
            return doc
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            raise

    def extract_text_blocks(self, pdf_doc):
        """
        Extracts text blocks with their bounding boxes, font info, and page number
        from a PDF document using pypdfium2.
        Each text block represents a contiguous piece of text with consistent formatting.
        """
        all_text_blocks = []
        for i, page in enumerate(pdf_doc):
            try:
                text_page = page.get_textpage()
                # Iterate through blocks (often lines or paragraphs)
                for block in text_page.get_text_blocks():
                    bbox = block.get_bbox() # (x_min, y_min, x_max, y_max)
                    text = block.get_text()
                    font_info = block.get_fontinfo() # Returns dict with 'fontname', 'flags', 'size'
                    
                    if text.strip(): # Only add if there's actual text
                        all_text_blocks.append({
                            'page_idx': i,
                            'text': text,
                            'bbox': bbox,
                            'font_name': font_info['fontname'],
                            'font_size': font_info['size'],
                            'is_bold': bool(font_info['flags'] & pdfium.FONT_SERIF), # A heuristic for bold, not perfectly reliable
                            'is_italic': bool(font_info['flags'] & pdfium.FONT_ITALIC),
                            'origin_x': bbox[0],
                            'origin_y': bbox[1],
                            'width': bbox[2] - bbox[0],
                            'height': bbox[3] - bbox[1]
                        })
                page.close() # Release resources for the page
            except Exception as e:
                logger.warning(f"Error extracting text from page {i} of PDF: {e}")
                continue
        return all_text_blocks

    def close_pdf(self, pdf_doc):
        """Closes the PDF document and releases resources."""
        if pdf_doc:
            pdf_doc.close()
            logger.info("PDF document closed.")

# Helper for loading PDF and extracting text directly for other modules
def load_and_parse_pdf(pdf_path):
    parser = PDFParser()
    doc = parser.load_pdf(pdf_path)
    text_blocks = parser.extract_text_blocks(doc)
    parser.close_pdf(doc)
    return text_blocks