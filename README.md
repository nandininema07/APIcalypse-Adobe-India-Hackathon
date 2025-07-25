# PDF Outline Extractor – Challenge 1A Submission (Adobe India Hackathon 2025)

---

## Problem Statement

PDFs dominate our digital knowledge-sharing ecosystem, but their structure is opaque to machines. Adobe’s Challenge 1A invited participants to create a system that extracts a structured outline from PDFs — specifically the Title, and headings like H1, H2, H3 — and exports this in a standardized JSON format.

This task needed to be completed offline, within 10 seconds, using only the CPU, and with a ≤200MB model constraint.

---

## Demo Video

Click here to watch the full demo: [Demo Link](#)

---

## Our Approach

We initially experimented with the DocLayNet dataset using a DistilBERT model (60% accuracy). However, we transitioned to DocHieNet with a BERT-tiny (17–18MB) model which offered comparable accuracy, significantly lower memory, and faster inference times — fitting the hackathon constraints perfectly.

Key Highlights:
- Multilingual support: Works with both English and Chinese PDFs.
- Context-aware predictions using MultiHeadAttention and custom classification + linking models.
- Structure understanding through spatial and textual embedding fusion.
- Optimized for low memory and high generalization.

---

## Folder Structure

```
PDF-Outline-Extractor
├── pdf_input/        # Upload your PDFs here
├── pdf_output/       # Output JSONs will be saved here
├── json_input/       # Intermediate metadata and inputs for the model
├── src/              # All source files for preprocessing and inference
│   ├── model/
│   ├── pipeline/
│   └── utils/
├── Dockerfile        # Docker image definition
├── requirements.txt
└── README.md
```

---

## Setup & Execution

We follow the standard execution process defined by the Adobe team.

### Build Docker Image

```bash
docker build --platform linux/amd64 -t apicalypse-extractor:latest .
```

### Run Docker Container

```bash
docker run --rm \
  -v $(pwd)/pdf_input:/app/input \
  -v $(pwd)/pdf_output:/app/output \
  --network none \
  apicalypse-extractor:latest
```

All .pdf files inside the pdf_input/ folder will be processed. Corresponding .json files will be saved in pdf_output/.

---

## Output Format

Example JSON:

```json
{
  "title": "Sample PDF Title",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "Deep Learning", "page": 2 },
    { "level": "H3", "text": "Transformer Networks", "page": 3 }
  ]
}
```

---

## Performance

| Metric                | Result                       |
|-----------------------|------------------------------|
| Execution Time        | < 10 seconds per 50-page PDF |
| Model Size            | ~18MB (bert-tiny)            |
| Network Dependency    | No, completely offline       |
| Architecture          | linux/amd64 compatible       |
| Multilingual          | English + Chinese            |

---

## PDFs Handled

Our solution is robust to:

- Research papers (IEEE, ACM formats)
- Business and annual reports
- Academic textbooks
- PDFs with multiple columns
- Low-text pages (fallback OCR logic)
- Multilingual PDFs (limited to English + Chinese)

---

## Models, Libraries & Dataset

| Component           | Implementation                        |
|--------------------|----------------------------------------|
| Dataset            | DocHieNet                              |
| Model              | BERT-tiny + MultiHeadAttention         |
| Feature Extraction | Text embeddings + Box coordinates      |
| Frameworks         | PyTorch, pdfplumber                    |
| Libraries          | NumPy, scikit-learn, transformers      |

### Why json_input?

This intermediate preprocessing step generates rich metadata that helps:
- Reduce reliance on rule-based methods
- Retain contextual + spatial document structure
- Enhance accuracy through intrinsic properties like x-y positions, page metadata, and element IDs

---

## Testing & Evaluation

| Metric              | Result                        |
|---------------------|-------------------------------|
| Training Split      | 70%                           |
| Validation Split    | 10%                           |
| Testing Split       | 20%                           |
| Accuracy (BERT-tiny)| Comparable to DistilBERT      |
| Generalization      | Strong (low bias + variance)  |

We evaluated on a diverse set of 25 PDFs and compared accuracy against known ground-truth structures. Our model consistently avoided overfitting and retained hierarchy-level accuracy.

---

## Team

- Nandini Nema
- Soham Chandane
- Parv Siria

---

## Conclusion

This project enables structured PDF intelligence at scale — offline, multilingual, and under strict resource constraints. With title/heading extraction and hierarchy detection, this is a plug-and-play module for semantic search, summarization, and document interlinking in future stages of the hackathon.

We’re ready for Round 1B.

---

## Repo Access

This repository is currently private as per hackathon rules. Will be made public once permitted.

---
