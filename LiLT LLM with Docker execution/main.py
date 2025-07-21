# main.py - Entry point
import os
import json
import logging
import time
from pathlib import Path
from src.hybrid_processor import HybridPDFProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    # Use environment variables for input/output paths in Docker
    input_dir_path = os.getenv("INPUT_DIR", "/app/input")
    output_dir_path = os.getenv("OUTPUT_DIR", "/app/output")

    input_dir = Path(input_dir_path)
    output_dir = Path(output_dir_path)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    try:
        processor = HybridPDFProcessor()
    except Exception as e:
        logger.critical(f"Failed to initialize HybridPDFProcessor: {e}")
        # Exit if essential components (like config) cannot be loaded
        return

    # Process all PDFs in input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in input directory: {input_dir}")
        return

    for pdf_file in pdf_files:
        start_time = time.time()
        
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Extract structure
            result = processor.extract_structure(str(pdf_file))
            
            # Save result
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=processor.config.get('output', {}).get('indent', 2), ensure_ascii=False)
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {pdf_file.name} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")
            # Create error output
            error_result = {
                "title": "Error",
                "outline": [],
                "error": str(e)
            }
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=processor.config.get('output', {}).get('indent', 2))

if __name__ == "__main__":
    main()