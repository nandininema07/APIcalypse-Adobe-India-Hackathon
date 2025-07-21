# src/synthetic_data_generator.py (Part 1: Content)
from faker import Faker
import random

# Initialize Faker with Hindi locale
fake_hi = Faker('hi_IN')

def generate_hindi_sentence():
    """Generates a random Hindi sentence."""
    return fake_hi.sentence(nb_words=random.randint(5, 15), variable_nb_words=True)

def generate_hindi_paragraph():
    """Generates a random Hindi paragraph."""
    return fake_hi.paragraph(nb_sentences=random.randint(3, 7), variable_nb_sentences=True)

# Example usage:
# print(generate_hindi_sentence())
# print(generate_hindi_paragraph())

# src/synthetic_data_generator.py (Part 2: Layout Simulation)

HEADING_TYPES = {
    "TITLE": {"font_sizes": [28, 32, 36], "is_bold": True, "is_centered": True, "y_spacing": "large"},
    "H1": {"font_sizes": [20, 22, 24], "is_bold": True, "is_centered": False, "y_spacing": "large"},
    "H2": {"font_sizes": [16, 18], "is_bold": True, "is_centered": False, "y_spacing": "medium"},
    "H3": {"font_sizes": [14, 15], "is_bold": True, "is_centered": False, "y_spacing": "small"},
    "BODY_TEXT": {"font_sizes": [10, 11, 12], "is_bold": False, "is_centered": False, "y_spacing": "tiny"}
}

# Define ranges for numerical features for more granularity
FONT_SIZE_RANGES = {
    "large": (20, 36),   # For titles, H1
    "medium": (14, 19),  # For H2, H3
    "small": (10, 13),   # For body text
    "extra_small": (8, 9) # For footers, captions (can be used for negative examples)
}

Y_SPACING_THRESHOLDS = { # Relative 'gap' above a block
    "tiny": (0.01, 0.02),  # Minimal spacing
    "small": (0.02, 0.05), # Small paragraph break
    "medium": (0.05, 0.1), # Heading break
    "large": (0.1, 0.2)    # Major section/title break
}

INDENTATION_THRESHOLDS = { # X-offset from left margin
    "none": (0.0, 0.01),
    "slight": (0.01, 0.03), # For sub-headings, lists
    "medium": (0.03, 0.06), # For list items, blockquotes
    "large": (0.06, 0.1)
}


def generate_layout_features(heading_type, introduce_inconsistency=False):
    """
    Generates realistic or inconsistent layout features for a text block.
    """
    features = {
        "font_size": None,
        "is_bold": HEADING_TYPES[heading_type]["is_bold"],
        "is_italic": False, # Keep simple for now
        "is_centered": HEADING_TYPES[heading_type]["is_centered"],
        "y_spacing": random.uniform(*Y_SPACING_THRESHOLDS[HEADING_TYPES[heading_type]["y_spacing"]]),
        "x_indentation": random.uniform(*INDENTATION_THRESHOLDS["none"]) # Default no indentation for headings/body
    }

    # Assign font size based on type's general range
    if heading_type == "TITLE":
        features["font_size"] = random.choice(HEADING_TYPES["TITLE"]["font_sizes"])
    elif heading_type == "H1":
        features["font_size"] = random.choice(HEADING_TYPES["H1"]["font_sizes"])
    elif heading_type == "H2":
        features["font_size"] = random.choice(HEADING_TYPES["H2"]["font_sizes"])
    elif heading_type == "H3":
        features["font_size"] = random.choice(HEADING_TYPES["H3"]["font_sizes"])
    else: # BODY_TEXT
        features["font_size"] = random.choice(HEADING_TYPES["BODY_TEXT"]["font_sizes"])

    # --- Introduce Intentional Inconsistencies (The "Unconventional" Part) ---
    if introduce_inconsistency and random.random() < 0.2: # 20% chance to introduce inconsistency
        if heading_type == "BODY_TEXT":
            # Make a body text look like a heading (larger font, bold)
            features["font_size"] = random.choice(HEADING_TYPES["H3"]["font_sizes"]) # Can be similar to H3
            features["is_bold"] = True
            features["y_spacing"] = random.uniform(*Y_SPACING_THRESHOLDS["medium"]) # Larger gap
        elif heading_type in ["H2", "H3"] and random.random() < 0.5:
            # Make a heading look like body text (smaller font, not bold)
            features["font_size"] = random.choice(HEADING_TYPES["BODY_TEXT"]["font_sizes"])
            features["is_bold"] = False
            features["y_spacing"] = random.uniform(*Y_SPACING_THRESHOLDS["small"]) # Smaller gap

    return features

def encode_features_for_fasttext(text_content, features):
    """
    Encodes layout features as special tokens prepended to the text content.
    """
    encoded_tokens = []

    # Font Size (binned)
    if features["font_size"] >= FONT_SIZE_RANGES["large"][0]:
        encoded_tokens.append("<FS_LARGE>")
    elif features["font_size"] >= FONT_SIZE_RANGES["medium"][0]:
        encoded_tokens.append("<FS_MEDIUM>")
    elif features["font_size"] >= FONT_SIZE_RANGES["small"][0]:
        encoded_tokens.append("<FS_SMALL>")
    else:
        encoded_tokens.append("<FS_XSMALL>") # For very small fonts (e.g., captions)

    # Boolean flags
    if features["is_bold"]:
        encoded_tokens.append("<BOLD>")
    if features["is_italic"]:
        encoded_tokens.append("<ITALIC>")
    if features["is_centered"]:
        encoded_tokens.append("<CENTERED>")

    # Y-Spacing (binned)
    if features["y_spacing"] >= Y_SPACING_THRESHOLDS["large"][0]:
        encoded_tokens.append("<LGAP>")
    elif features["y_spacing"] >= Y_SPACING_THRESHOLDS["medium"][0]:
        encoded_tokens.append("<MGAP>")
    elif features["y_spacing"] >= Y_SPACING_THRESHOLDS["small"][0]:
        encoded_tokens.append("<SGAP>")
    else:
        encoded_tokens.append("<TGAP>") # Tiny gap

    # X-Indentation (binned)
    if features["x_indentation"] >= INDENTATION_THRESHOLDS["large"][0]:
        encoded_tokens.append("<LINDENT>")
    elif features["x_indentation"] >= INDENTATION_THRESHOLDS["medium"][0]:
        encoded_tokens.append("<MINDENT>")
    elif features["x_indentation"] >= INDENTATION_THRESHOLDS["slight"][0]:
        encoded_tokens.append("<SINDENT>")
    else:
        encoded_tokens.append("<NINDENT>") # No indentation

    # Add other features like is_list_item if generated
    # if features.get("is_list_item", False):
    #     encoded_tokens.append("<BULLET>")


    return " ".join(encoded_tokens + [text_content.strip()])

# src/synthetic_data_generator.py (Part 3: Document Generation)

def generate_document_section(max_depth=3, current_depth=0):
    """Generates a hierarchical section of a document."""
    sections = []

    if current_depth == 0: # This is the document root or first page
        # Title
        title_text = fake_hi.sentence(nb_words=random.randint(4, 8)).title()
        title_features = generate_layout_features("TITLE")
        sections.append(f"__label__TITLE {encode_features_for_fasttext(title_text, title_features)}")

    # H1 sections
    num_h1 = random.randint(1, 3)
    for _ in range(num_h1):
        h1_text = generate_hindi_sentence().title() # Use title case for headings
        h1_features = generate_layout_features("H1")
        sections.append(f"__label__H1 {encode_features_for_fasttext(h1_text, h1_features)}")

        # Body text under H1
        for _ in range(random.randint(1, 3)):
            body_text = generate_hindi_paragraph()
            body_features = generate_layout_features("BODY_TEXT", introduce_inconsistency=True) # Introduce inconsistencies here
            sections.append(f"__label__BODY_TEXT {encode_features_for_fasttext(body_text, body_features)}")

        if current_depth < max_depth -1: # Allow for nested H2/H3
            # H2 sections
            num_h2 = random.randint(0, 2)
            for _ in range(num_h2):
                h2_text = generate_hindi_sentence().title()
                h2_features = generate_layout_features("H2")
                sections.append(f"__label__H2 {encode_features_for_fasttext(h2_text, h2_features)}")

                # Body text under H2
                for _ in range(random.randint(1, 2)):
                    body_text = generate_hindi_paragraph()
                    body_features = generate_layout_features("BODY_TEXT", introduce_inconsistency=True)
                    sections.append(f"__label__BODY_TEXT {encode_features_for_fasttext(body_text, body_features)}")

                if current_depth < max_depth -2: # Allow for nested H3
                    # H3 sections
                    num_h3 = random.randint(0, 2)
                    for _ in range(num_h3):
                        h3_text = generate_hindi_sentence().title()
                        h3_features = generate_layout_features("H3")
                        sections.append(f"__label__H3 {encode_features_for_fasttext(h3_text, h3_features)}")

                        # Body text under H3
                        for _ in range(random.randint(1, 2)):
                            body_text = generate_hindi_paragraph()
                            body_features = generate_layout_features("BODY_TEXT", introduce_inconsistency=True)
                            sections.append(f"__label__BODY_TEXT {encode_features_for_fasttext(body_text, body_features)}")

    return sections

def generate_synthetic_hindi_dataset(num_documents=100, output_file="data/synthetic/synthetic_hi.txt"):
    """Generates a synthetic Hindi dataset."""
    print(f"Generating {num_documents} synthetic Hindi documents...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_documents):
            document_sections = generate_document_section(max_depth=3, current_depth=0)
            for line in document_sections:
                f.write(line + "\n")
            if i < num_documents - 1:
                f.write("\n") # Add a blank line to separate documents conceptually
    print(f"Synthetic Hindi dataset saved to {output_file}")

# Example usage (run this once to generate the dataset):
# generate_synthetic_hindi_dataset(num_documents=500) # Generate 500 synthetic documents

# src/synthetic_data_generator.py (at the very end of the file)

if __name__ == "__main__":
    # Create the data/synthetic directory if it doesn't exist
    import os
    output_dir = "data/synthetic"
    os.makedirs(output_dir, exist_ok=True)

    # Generate 500 synthetic Hindi documents. You can adjust this number.
    # More documents mean more training data, but also longer generation time.
    generate_synthetic_hindi_dataset(num_documents=500, output_file=os.path.join(output_dir, "synthetic_hi.txt"))

    print("\nSynthetic Hindi dataset generation complete!")
    print(f"Check the file: {os.path.join(output_dir, 'synthetic_hi.txt')}")