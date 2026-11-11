
import os
import re
import math
import tempfile

import fitz
import numpy as np
import networkx as nx

from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "text-summarizer-secret")

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}


# ---------------- PDF TEXT EXTRACTION ----------------

def extract_pdf_text(pdf_path):

    text = []

    try:
        doc = fitz.open(pdf_path)

        if doc.is_encrypted:
            doc.close()
            raise ValueError("Password-protected PDFs are not supported.")

        for page in doc:
            page_text = page.get_text("text")
            if page_text.strip():
                text.append(page_text)

        doc.close()

    except ValueError:
        raise
    except Exception as e:
        raise ValueError("Unable to read PDF file.") from e

    return "\n".join(text).strip()


# ---------------- TEXTRANK SUMMARIZATION ----------------

def textrank_summarize(article, summary_length="medium"):

    # Step 1: Sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', article.strip())

    sentences = [
        s.strip() for s in sentences if s.strip()
    ]

    if not sentences:
        return "No article text provided."

    if len(sentences) <= 2:
        return " ".join(sentences)

    # Step 2: Medium summary selection
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

    num_sentences = min(num_sentences, len(sentences))

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Step 3: TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        sentence_vectors = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:num_sentences])

    # Step 4: Cosine Similarity
    similarity_matrix = cosine_similarity(sentence_vectors)

    # Step 5: Remove self-connections
    np.fill_diagonal(similarity_matrix, 0)

    # Step 6: Create weighted graph
    graph = nx.from_numpy_array(similarity_matrix)

    # Step 7: Apply PageRank
    scores = nx.pagerank(graph, weight="weight")

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
        sentences[i] for i in selected_indices
    )

    return summary


# ---------------- WORD COUNT ----------------

def word_count(text):
    return len(re.findall(r"\b[\w'-]+\b", text))


# ---------------- FILE VALIDATION ----------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ---------------- HOME PAGE ----------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":

        file = request.files.get("article")

        if not file or file.filename == "":
            flash("Please upload a PDF document.")
            return render_template("index.html")

        filename = secure_filename(file.filename)

        if not allowed_file(filename):
            flash("Only PDF files are supported.")
            return render_template("index.html")

        temp_path = None

        try:

            # Save temporary PDF
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as temp:
                temp_path = temp.name
                file.save(temp_path)

            # Extract text
            article_text = extract_pdf_text(temp_path)

            if not article_text:
                flash("No readable text found in PDF.")
                return render_template("index.html")

            # Generate summary
            summary = textrank_summarize(
                article_text,
                summary_length="medium"
            )

            # Statistics
            original_words = word_count(article_text)
            summary_words = word_count(summary)

            compression = 0

            if original_words:
                compression = round(
                    (1 - summary_words / original_words) * 100,
                    2
                )

            result = {
                "filename": filename,
                "original_text": article_text,
                "summary": summary,
                "original_words": original_words,
                "summary_words": summary_words,
                "compression": compression
            }

        except ValueError as e:
            flash(str(e))

        except Exception:
            app.logger.exception("PDF processing error")
            flash("Something went wrong while processing the PDF.")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return render_template("index.html", result=result)


# ---------------- FILE SIZE ERROR ----------------

@app.errorhandler(413)
def file_too_large(error):

    flash("PDF size must be less than 15 MB.")

    return render_template("index.html"), 413


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
