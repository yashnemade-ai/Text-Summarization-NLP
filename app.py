
import os
import re
import math
import tempfile
import logging

import fitz
import requests
import numpy as np
import networkx as nx
import nltk

from nltk import wordpunct_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename

# ---------------- FLASK CONFIGURATION ----------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "text-summarizer-secret"
)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "webp"
}

OCR_API_URL = "https://api.ocr.space/parse/image"

logging.basicConfig(level=logging.INFO)

# ---------------- NLTK RESOURCES ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")

nltk.data.path.insert(0, NLTK_DATA_DIR)

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):

    if tag.startswith("J"):
        return wordnet.ADJ

    elif tag.startswith("V"):
        return wordnet.VERB

    elif tag.startswith("N"):
        return wordnet.NOUN

    elif tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN

# ---------------- FILE VALIDATION ----------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# ---------------- OCR SPACE API ----------------

def extract_ocr_text(file_bytes, filename):

    api_key = os.environ.get("OCR_API_KEY")

    if not api_key:
        raise ValueError(
            "OCR API key is missing. "
            "Please configure OCR_API_KEY in Render."
        )

    try:

        response = requests.post(
            OCR_API_URL,

            files={
                "file": (
                    filename,
                    file_bytes,
                    "application/octet-stream"
                )
            },

            data={
                "apikey": api_key,
                "language": "eng",
                "isOverlayRequired": "false"
            },

            timeout=90
        )

        response.raise_for_status()

        result = response.json()

        if result.get("IsErroredOnProcessing"):

            error = result.get("ErrorMessage", "OCR failed")

            raise ValueError(str(error))

        pages = result.get("ParsedResults", [])

        extracted_text = "\n".join(
            page.get("ParsedText", "")
            for page in pages
        )

        return extracted_text.strip()

    except requests.RequestException as e:

        app.logger.exception("OCR API request failed")

        raise ValueError(
            "OCR service is unavailable. Please try again."
        ) from e

# ---------------- IMAGE TEXT EXTRACTION ----------------

def extract_image_text(image_path):

    with open(image_path, "rb") as image_file:

        image_bytes = image_file.read()

    filename = os.path.basename(image_path)

    return extract_ocr_text(
        image_bytes,
        filename
    )

# ---------------- PDF TEXT EXTRACTION ----------------

def extract_pdf_text(pdf_path):

    extracted_pages = []

    try:

        with fitz.open(pdf_path) as doc:

            if doc.is_encrypted:

                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            for page_number, page in enumerate(doc):

                # First try normal PDF text extraction
                page_text = page.get_text("text").strip()

                if page_text:

                    extracted_pages.append(page_text)

                else:

                    # OCR fallback for scanned PDF pages
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False
                    )

                    image_bytes = pix.tobytes("png")

                    ocr_text = extract_ocr_text(
                        image_bytes,
                        f"page_{page_number + 1}.png"
                    )

                    if ocr_text:
                        extracted_pages.append(ocr_text)

    except ValueError:
        raise

    except Exception as e:

        app.logger.exception("PDF extraction failed")

        raise ValueError(
            "Unable to read the PDF document."
        ) from e

    return "\n\n".join(extracted_pages).strip()

# ---------------- POS TAGGING + LEMMATIZATION ----------------

def preprocess_sentence(sentence):

    # Step 1: Tokenization
    words = wordpunct_tokenize(sentence)

    # Step 2: POS Tagging
    tagged_words = pos_tag(words)

    processed_words = []

    # Step 3: Lemmatization
    for word, tag in tagged_words:

        if word.isalpha():

            pos = get_wordnet_pos(tag)

            lemma = lemmatizer.lemmatize(
                word.lower(),
                pos
            )

            processed_words.append(lemma)

    return processed_words

# ---------------- JACCARD SIMILARITY ----------------

def jaccard_similarity(sentence1, sentence2):

    set1 = set(sentence1)
    set2 = set(sentence2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))

    union = len(set1.union(set2))

    return intersection / union

# ---------------- TEXTRANK SUMMARIZATION ----------------

def textrank_summarize(article, summary_length="medium"):

    # Step 1: Sentence Tokenization
    sentences = re.split(
        r'(?<=[.!?])\s+',
        article.strip()
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return "No article text provided."

    if len(sentences) <= 2:
        return " ".join(sentences)

    # Step 2: Summary Length
    ratios = {
        "short": 0.15,
        "medium": 0.30,
        "long": 0.50
    }

    ratio = ratios.get(summary_length, 0.30)

    num_sentences = max(
        2,
        math.ceil(len(sentences) * ratio)
    )

    num_sentences = min(
        num_sentences,
        len(sentences)
    )

    # Step 3: POS Tagging + Lemmatization
    processed_sentences = [
        preprocess_sentence(sentence)
        for sentence in sentences
    ]

    # Step 4: Create Similarity Matrix using Jaccard
    sentence_count = len(sentences)

    similarity_matrix = np.zeros(
        (sentence_count, sentence_count)
    )

    for i in range(sentence_count):

        for j in range(i + 1, sentence_count):

            similarity = jaccard_similarity(
                processed_sentences[i],
                processed_sentences[j]
            )

            similarity_matrix[i][j] = similarity

            similarity_matrix[j][i] = similarity

    # Step 5: Remove Self Connections
    np.fill_diagonal(
        similarity_matrix,
        0
    )

    # Step 6: Create Weighted Graph
    graph = nx.from_numpy_array(
        similarity_matrix
    )

    # Step 7: Apply PageRank
    scores = nx.pagerank(
        graph,
        weight="weight"
    )

    # Step 8: Rank Sentences
    ranked_sentences = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    # Step 9: Select Top Sentences
    selected_indices = sorted(
        ranked_sentences[:num_sentences]
    )

    # Step 10: Generate Summary in Original Order
    summary = " ".join(
        sentences[i]
        for i in selected_indices
    )

    return summary

# ---------------- WORD COUNT ----------------

def word_count(text):

    return len(
        re.findall(r"\b[\w'-]+\b", text)
    )

# ---------------- HOME PAGE ----------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":

        file = request.files.get("article")

        if not file or file.filename == "":

            flash("Please upload a PDF or image.")

            return render_template("index.html")

        filename = secure_filename(file.filename)

        if not filename or not allowed_file(filename):

            flash(
                "Supported formats: PDF, PNG, JPG, JPEG and WEBP."
            )

            return render_template("index.html")

        extension = filename.rsplit(".", 1)[1].lower()

        temp_path = None

        try:

            # Save temporary file
            with tempfile.NamedTemporaryFile(
                suffix="." + extension,
                delete=False
            ) as temp:

                temp_path = temp.name

                file.save(temp_path)

            # Extract text based on file type
            if extension == "pdf":

                article_text = extract_pdf_text(
                    temp_path
                )

                file_type = "PDF Document"

            else:

                article_text = extract_image_text(
                    temp_path
                )

                file_type = "Image"

            if not article_text.strip():

                flash(
                    "No readable text found. "
                    "Please upload a clearer document."
                )

                return render_template("index.html")

            # Generate TextRank summary
            summary = textrank_summarize(
                article_text,
                summary_length="medium"
            )

            # Calculate statistics
            original_words = word_count(
                article_text
            )

            summary_words = word_count(
                summary
            )

            compression = 0

            if original_words > 0:

                compression = round(
                    (
                        1 - summary_words / original_words
                    ) * 100,
                    2
                )

            result = {

                "filename": filename,

                "file_type": file_type,

                "original_text": article_text,

                "summary": summary,

                "original_words": original_words,

                "summary_words": summary_words,

                "compression": compression
            }

        except ValueError as e:

            flash(str(e))

        except Exception:

            app.logger.exception(
                "Document processing error"
            )

            flash(
                "Something went wrong while processing the document."
            )

        finally:

            if temp_path and os.path.exists(temp_path):

                os.remove(temp_path)

    return render_template(
        "index.html",
        result=result
    )

# ---------------- FILE SIZE ERROR ----------------

@app.errorhandler(413)
def file_too_large(error):

    flash(
        "File size must be less than 15 MB."
    )

    return render_template("index.html"), 413

# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
