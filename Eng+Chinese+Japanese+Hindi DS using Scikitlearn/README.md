# 

We must strictly adhere to the constraints (CPU-only, ≤200MB model, ≤10 seconds) and the requirement to handle "complicated PDFs" that break simple font rules. Our current dataset strategy for Round 1A is precisely designed for this:

- DocHieNet (English & Chinese for Hierarchy): This dataset provides explicit annotations for hierarchical document structure and layout entities, which are directly mappable to Title, H1, H2, H3.

- HJDataset (Japanese for Complex Layouts & Hierarchy): This is crucial for the Japanese bonus, offering hierarchical structures and detailed layout element annotations in complex historical Japanese documents.

- Synthetic Data Generation (for Robustness & Hindi/Devanagari): This is our "unconventional" ace. By programmatically generating PDFs (or PDF-like data) with known heading structures and intentionally inconsistent layouts (e.g., a bold body text that looks like a heading), we can create an unlimited, perfectly labeled dataset that forces our model to learn genuine layout patterns beyond simple font rules. This is also the most efficient way to introduce Hindi/Devanagari layout examples without relying on scarce public datasets.

These datasets, combined with PyMuPDF for extracting granular layout features, will provide the necessary visual and structural cues for our FastText (or NeuroBERT-Mini) classifier to accurately identify headings.

Future Consideration for Round 1B (Semantic Understanding)
While we are strictly focusing on Round 1A now, it's worth noting that MTOP's multilingual utterances and parallel structure could be relevant for Round 1B's semantic understanding tasks (e.g., fine-tuning multilingual embeddings for persona-driven relevance ranking), but that's a discussion for when we reach Phase 1B.

## Hindi Synthetic Data Generation
Goal
Generate a text file (e.g., synthetic_hi.txt) where each line represents a synthetic text block from a document, formatted for FastText training:
__label__HEADING_LEVEL <FEATURE_TOKEN_1> <FEATURE_TOKEN_2> ... <FEATURE_TOKEN_N> Actual Hindi Text Content

_______________________________________________________________________

Now, let's move to the implementation phase, starting with the core of our "Unconventional Precision" approach: src/pdf_parser.py. This module will be responsible for extracting the raw, granular layout information from PDF files using PyMuPDF, forming the foundation for all subsequent feature engineering.

## Phase 1A Implementation: Part 1 - src/pdf_parser.py
Objective: To extract all text and its associated layout metadata (bounding box, font details, page number) from PDF documents. This script acts as the lowest-level interface with the PDF binary format.

Key Library: PyMuPDF (specifically fitz module)

### Explanation & Key Points:
extract_page_elements(pdf_path) function:
- Takes the path to a single PDF file as input.
- Opens the PDF using fitz.open(pdf_path).
- Iterates through each page in the document.
- page.get_text("rawdict"): This is the crucial part. It instructs PyMuPDF to return a detailed dictionary representation of the page content. This rawdict contains blocks, lines, and spans, providing granular information including bounding boxes (bbox), font names (font), font sizes (size), and flags (which encode bold/italic/monospace status). This is our source of "unconventional precision."
- Span-Level Extraction: For simplicity and direct relevance to headings, we're primarily extracting information at the span level. A span is a contiguous string of text that shares the same font, size, and style. Headings are often single spans or a few spans on a line.
- Font Flags: We use bitwise operations (&) with fitz.TEXT_FONT_BOLD and fitz.TEXT_FONT_ITALIC to correctly determine if a span is bold or italic.
- Output: The function returns a List[Dict] where each dictionary represents an extracted span with its text and rich metadata.

Example Usage (if __name__ == "__main__":)
- This block demonstrates how to run the extract_page_elements function.
- Important: You must replace "data/raw/sample.pdf" with the actual path to a PDF you want to test.
- I've included a small try-except block to create a very basic dummy PDF if sample.pdf isn't found, just so you can run the script and see some output immediately. For real development, use actual PDFs from your DocHieNet, HJDataset, or the hackathon's sample.pdf.
- The output prints the first few extracted elements, showing their text, bounding box, font info, and bold/italic status. This will help you verify that the extraction is working correctly.

## Phase 1A Implementation: Part 2 - src/feature_extractor.py
Objective: To take the word-level elements from pdf_parser.py, group them into logical lines and text blocks, compute advanced layout features (indentation, spacing, relative font sizes), and prepare the data in the __label__HEADING_LEVEL <FEATURE_TOKENS> Actual Text format for FastText.

### Implement src/feature_extractor.py
1. pdf_parser.py is working (with fallback):
- DEBUG: Calculated absolute path... Using existing PDF...
- DEBUG: Successfully opened PDF. Page count: 14
- DEBUG: Page X: 'rawdict' yielded no valid text. Attempting fallback to 'words'. (This confirms rawdict still doesn't work for your system, which we've accepted as a limitation for now).
- DEBUG: Page X: Fallback 'words' extraction found Y words. (This is the critical success! It means pdf_parser.py is successfully extracting words and their bounding boxes using the more robust get_text("words") method.)

2. feature_extractor.py is working:
- DEBUG: Successfully extracted 4630 raw elements. Grouping into lines...
- DEBUG: Grouped into 474 lines. Grouping into blocks...
- DEBUG: Grouped into 40 blocks. Extracting advanced features...
- DEBUG: Extracted advanced features for 40 blocks.
This indicates that your group_elements_into_lines, group_lines_into_blocks, and extract_features_from_blocks functions are successfully processing the word-level data.
3. Features are being extracted and encoded for FastText:
The --- Processed Blocks with Encoded Features (First 10) --- section shows exactly what we aimed for:
- Approx Font Size, Relative Y-Spacing, X-Indentation are calculated.
- Raw Text is extracted.
- The FastText Input line shows the encoded features (e.g., <FS_LARGE>, <LGAP>, <INDENT>) prepended to the cleaned text, formatted as __label__DUMMY_LABEL .... This confirms the feature encoding is working!

### Implement src/data_processor.py
1. Synthetic Hindi Data (SUCCESS):
- INFO: Added 8963 lines from synthetic Hindi data.
- This is perfect. Your synthetic data generation and its integration are working.

2. DocHieNet & HJDataset Processing (Placeholders Acknowledged):
- INFO: Starting DocHieNet processing (Conceptual - requires full JSON parsing).
- DEBUG: Processing DocHieNet from ... (returning dummy data for now).
- INFO: DocHieNet processing placeholder complete.
- Same for HJDataset. This is expected. The script is correctly using the dummy data for now as per the placeholder functions.

3. Sample PDF Processing (MAJOR PROGRESS!):
- INFO: Processing sample.pdf using its ground truth JSON (sample.json).
- DEBUG: Calculated absolute path... Using existing PDF... Successfully opened PDF. Page count: 14
- DEBUG: Page X: 'rawdict' yielded no valid text. Attempting fallback to 'words'.
- DEBUG: Page X: Fallback 'words' extraction found Y words.
This confirms:
- The absolute path resolution works.
- pdf_parser.py is successfully opening and extracting thousands of words from your 14-page sample.pdf using the robust "words" fallback.
- feature_extractor.py is successfully processing these words into lines and blocks.

4. Sample PDF Matching (Expected Warnings & Next Refinement Point):
- You're seeing WARNING: Sample PDF: Could not find a good match for GT entry: ...
- This is expected and is the next logical step to refine. The matching logic implemented for sample.pdf in data_processor.py is currently very simplistic (a basic substring check). Given the complexity of the sample.pdf (it has headers like RFP: Request for Proposal and The Ontario Digital Library split across multiple lines and blocks in the PDF itself), a simple substring match won't always work perfectly. The E0H1CM114.json you provided shows H1 entries like:
    - "Ontario’s Digital Library " 
    - "A Critical Component for Implementing Ontario’s Road Map to Prosperity Strategy " 
    - Your pdf_parser might extract "Ontario's", "Digital", "Library" as separate words or groups that don't directly match the exact GT string "Ontario's Digital Library" in one block. Also, the title is "RFP:Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library"  which is clearly composed of many extracted blocks from pdf_parser.
    - This means our current block-grouping and matching logic needs refinement.

5. Output Files (SUCCESS!):
- SUCCESS: Training data written to ...train.txt (7221 lines).
- SUCCESS: Validation data written to ...val.txt (896 lines).
- SUCCESS: Test data written to ...test.txt (897 lines).
- This is the ultimate success! You now have your FastText-ready training, validation, and test data files generated in data/processed/heading_classification/. They contain thousands of lines of synthetic Hindi data, plus some data from sample.pdf.

#### We have achieved the primary goal of Phase 0: You can now generate the necessary data files for model training!
The pdf_parser.py is working, feature_extractor.py is working, and data_processor.py is orchestrating the creation of the final training files.

## Phase 1A Implementation: Part 3 - src/heading_classifier.py (FastText Training)
Objective: Train a FastText model using the generated train.txt to classify text blocks into Title, H1, H2, H3, BODY_TEXT.
FastText not working on windows, will be directly installed in Dockerfile in Linux evn

1. Your local development environment is now fully functional. You successfully executed src/heading_classifier.py.

2. The placeholder model trained and evaluated successfully.
- INFO: Starting Placeholder Model training...
- INFO: Loaded 6819 training samples.
- INFO: Placeholder model training complete. Saving model...
- INFO: Model saved.

3. The model metrics are excellent:
- Validation Results: Overall accuracy of 0.97, with H1, H3, and TITLE classes achieving 1.00 Precision, Recall, and F1-score. H2 has a respectable 0.85 F1-score.
- Test Results: Similarly strong performance with overall accuracy of 0.96.
- This indicates that your synthetic Hindi data (which is a large portion of your training data) and the basic matching for sample.pdf are structured well enough for the scikit-learn model to learn.

4. Model Size is well within constraints:
- INFO: Trained placeholder model size: 0.88 MB
- INFO: Placeholder model size is within the <= 200MB constraint for Round 1A.
- This confirms the efficiency of our chosen placeholder approach.

## Phase 1A Implementation: Part 4 - src/outline_reconstructor.py
Objective: To take a PDF document, process it through the parser and feature extractor, predict heading levels using the trained model, and then reconstruct a clean, hierarchical JSON outline as per the hackathon requirements.
