import os
import fitz  # PyMuPDF

def compress_pdf(input_path, output_path, max_size_kb=1000, quality=80):
    """
    Compress a PDF file to be under max_size_kb.
    Only compresses if input file is larger than max_size_kb.
    Returns True if compression was performed, False otherwise.
    """
    input_size_kb = os.path.getsize(input_path) // 1024
    if input_size_kb <= max_size_kb:
        # No compression needed
        return False

    doc = fitz.open(input_path)
    # Recompress images in each page
    for page in doc:
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            # Re-encode image with lower quality if JPEG, else skip
            if image_ext.lower() in ["jpeg", "jpg"]:
                from PIL import Image
                import io
                pil_img = Image.open(io.BytesIO(image_bytes))
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format="JPEG", quality=quality)
                new_img_bytes = img_buffer.getvalue()
                doc.update_stream(xref, new_img_bytes)
    # Save with garbage collection and deflate
    doc.save(
        output_path,
        garbage=4,
        deflate=True,
        clean=True,
    )
    doc.close()
    # Check if output is now under max_size_kb
    output_size_kb = os.path.getsize(output_path) // 1024
    if output_size_kb > max_size_kb:
        print(f"Warning: Compressed file still exceeds {max_size_kb}KB.")
    return True