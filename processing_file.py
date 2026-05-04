import logging
import io
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import nltk
from nltk.tokenize import word_tokenize

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO
)

# ── Constants ──────────────────────────────────────────────
MAX_DOCS = 3
MAX_PAGES_PER_DOC = 5
MAX_TOKENS_APPROX = 3000

# ── Stopword Lists ─────────────────────────────────────────
SAFE_TO_REMOVE = {
    "a", "an", "the",
    "is", "are", "was", "were",
    "of", "in", "on", "at",
    "and", "or",
    "this", "that", "these",
    "it", "its"
}

PRESERVE = {
    "not", "no", "never", "without",
    "don't", "doesn't", "can't", "won't",
    "isn't", "aren't", "only", "must",
    "should", "always", "except"
}


# ── Stopword Removal ───────────────────────────────────────
def remove_safe_stopwords(text: str) -> str:
    """
    Remove only truly meaningless words from document text.
    Preserves sentence structure and negations.
    Saves ~20-30% tokens without losing meaning.
    """
    words = word_tokenize(text)
    filtered = [
        w for w in words
        if w.lower() not in SAFE_TO_REMOVE or w.lower() in PRESERVE
    ]
    return " ".join(filtered)


# ── PDF Parsing ────────────────────────────────────────────
def get_text_pdf(pdf_docs):
    """
    Extract text from up to MAX_DOCS PDFs, capped at MAX_PAGES_PER_DOC each.
    Returns list of dicts: [{"name": filename, "text": extracted_text}]
    """
    if len(pdf_docs) > MAX_DOCS:
        logging.warning(f"More than {MAX_DOCS} PDFs uploaded. Only first {MAX_DOCS} will be used.")
        pdf_docs = pdf_docs[:MAX_DOCS]

    documents = []
    for pdf in pdf_docs:
        text = ""
        try:
            pdf_reader = PdfReader(pdf)
            total_pages = len(pdf_reader.pages)
            pages_to_read = min(total_pages, MAX_PAGES_PER_DOC)

            if total_pages > MAX_PAGES_PER_DOC:
                logging.warning(
                    f"{pdf.name}: {total_pages} pages found. Only first {MAX_PAGES_PER_DOC} will be used."
                )

            for i in range(pages_to_read):
                page_text = pdf_reader.pages[i].extract_text() or ""
                text += f"\n[Page {i + 1}]\n{page_text}"

            documents.append({"name": pdf.name, "text": text.strip()})
            logging.info(f"Extracted text from {pdf.name} ({pages_to_read} pages)")

        except Exception as e:
            logging.error(f"Error reading PDF {getattr(pdf, 'name', 'file')}: {e}")

    return documents


# ── Image OCR ──────────────────────────────────────────────
def get_text_image(image_files):
    """
    Extract text from image files using Tesseract OCR.
    Returns list of dicts: [{"name": filename, "text": ocr_text}]
    """
    if len(image_files) > MAX_DOCS:
        logging.warning(f"More than {MAX_DOCS} images uploaded. Only first {MAX_DOCS} will be used.")
        image_files = image_files[:MAX_DOCS]

    documents = []
    for img_file in image_files:
        try:
            image = Image.open(io.BytesIO(img_file.read()))
            ocr_text = pytesseract.image_to_string(image)
            documents.append({"name": img_file.name, "text": ocr_text.strip()})
            logging.info(f"OCR extracted text from image: {img_file.name}")

        except Exception as e:
            logging.error(f"Error OCR-ing image {getattr(img_file, 'name', 'file')}: {e}")

    return documents


# ── Context Builder ────────────────────────────────────────
def build_context(documents):
    """
    Combine all document texts into a single structured context string.
    Step 1: Remove safe stopwords to save ~20-30% tokens
    Step 2: Apply word count proxy check against token budget
    Step 3: Truncate only if still too long after stopword removal
    """
    context_parts = []
    for doc in documents:
        header = f"=== Document: {doc['name']} ==="

        # Step 1 — remove safe stopwords to compress text
        cleaned_text = remove_safe_stopwords(doc["text"])
        logging.info(
            f"Stopword removal: {len(doc['text'].split())} → {len(cleaned_text.split())} words for {doc['name']}"
        )

        context_parts.append(f"{header}\n{cleaned_text}")

    full_context = "\n\n".join(context_parts)

    # Step 2 — word count proxy: 1 token ≈ 0.75 words
    word_budget = int(MAX_TOKENS_APPROX * 0.75)
    words = full_context.split()

    # Step 3 — truncate only if still over budget after stopword removal
    if len(words) > word_budget:
        logging.warning(
            f"Context still too long ({len(words)} words) after stopword removal. "
            f"Truncating to ~{word_budget} words."
        )
        full_context = " ".join(words[:word_budget]) + \
                       "\n\n[... content truncated due to token limit ...]"

    return full_context