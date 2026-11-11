
import os
import re
import math
import tempfile

import fitz
import numpy as np
import networkx as nx
import pytesseract

from PIL import Image, ImageOps
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY", "text-summarizer-secret"
)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "pdf", "png", "jpg", "jpeg", "webp"
}


# ---------------- FILE VALIDATION ----------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ---------------- IMAGE TEXT EXTRACTION ----------------

def extract_image_text(image_path):

    try:
        image = Image.open(image_path)

        # Correct image orientation
        image = ImageOps.exif_transpose(image)

        # Convert to RGB
        image = image.convert("RGB")

        # Extract text using OCR
        extracted_text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        return extracted_text.strip()

    except pytesseract.pytesseract.TesseractNotFoundError:
        raise ValueError(
            "Tesseract OCR is not installed on the server."
        )

    except Exception as e:
        raise ValueError(
            "Unable to extract text from this image."
        ) from e


# ---------------- PDF TEXT EXTRACTION ----------------

def extract_pdf_text(pdf_path):

    text = []

    try:

        with fitz.open(pdf_path) as doc:

            if doc.is_encrypted:
                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            for page in doc:

                page_text = page.get_text("text").strip()

                # Normal PDF text extraction
                if page_text:
                    text.append(page_text)

                else:
                    # OCR fallback for scanned PDF pages
                    try:
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(2, 2),
                            alpha=False
                        )

                        image = Image.frombytes(
                            "RGB",
                            [pix.width, pix.height],
                            pix.samples
                        )

                        ocr_text = pytesseract.image_to_string(
                            image,
                            lang="eng"
                        ).strip()

                        if ocr_text:
                            text.append(ocr_text)

                    except pytesseract.pytesseract.TesseractNotFoundError:
                        raise ValueError(
                            "This PDF is scanned. Tesseract OCR "
                            "is required to extract its text."
                        )

    except ValueError:
        raise

    except Exception as e:
        raise ValueError(
            "Unable to read this PDF file."
        ) from e

    return "\n".join(text).strip()


# ---------------- TEXTRANK SUMMARIZATION ----------------

def textrank_summarize(article, summary_length="medium"):

    # Step 1: Sentence splitting
    sentences = re.split(
        r'(?<=[.!?])\s+',
        article.strip()
    )

    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]

    if not sentences:
        return "No article text provided."

    if len(sentences) <= 2:
        return " ".join(sentences)

    # Step 2: Summary length
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

    # Step 3: TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:
        sentence_vectors = vectorizer.fit_transform(
            sentences
        )

    except ValueError:
        return " ".join(
            sentences[:num_sentences]
        )

    # Step 4: Cosine Similarity
    similarity_matrix = cosine_similarity(
        sentence_vectors
    )

    # Step 5: Remove self-connections
    np.fill_diagonal(
        similarity_matrix,
        0
    )

    # Step 6: Create weighted graph
    graph = nx.from_numpy_array(
        similarity_matrix
    )

    # Step 7: PageRank
    scores = nx.pagerank(
        graph,
        weight="weight"
    )

    # Step 8: Rank sentences
    ranked_sentences = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    selected_indices = sorted(
        ranked_sentences[:num_sentences]
    )

    # Step 9: Original sentence order
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

        if not allowed_file(filename):
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

            # Extract text according to file type
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
                    "Please upload a clearer document or image."
                )

                return render_template("index.html")

            # Generate summary using TextRank
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

            if original_words:

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
                "Something went wrong while processing the file."
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
