import html
import io
import os
import uuid

from flask import (
    Flask, redirect, render_template, request, send_file, session, url_for,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from werkzeug.utils import secure_filename

from extractor import extract_text_from_file
from summarizer import textrank_summarize

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}

# Length -> num_sentences (LAB7 uses num_sentences)
LENGTH_MAP = {"short": 3, "medium": 5, "long": 8}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    original = ""
    result = None
    error = None
    length = "medium"
    uploaded_name = None

    if request.method == "POST":
        length = request.form.get("length", "medium")
        text = request.form.get("text", "").strip()
        file = request.files.get("file")

        try:
            if file and file.filename:
                if not allowed_file(file.filename):
                    raise ValueError("Sirf PDF, JPG, PNG files allowed hain.")

                original_name = secure_filename(file.filename)
                uploaded_name = original_name
                unique_name = f"{uuid.uuid4().hex}_{original_name}"
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                file.save(save_path)

                text = extract_text_from_file(save_path)

            if not text or len(text.strip()) < 20:
                raise ValueError(
                    "Text extract nahi hua ya bahut chhota hai. Doosri file ya zyada text try karo."
                )

            original = text.strip()
            num_sentences = LENGTH_MAP.get(length, 5)
            result = textrank_summarize(original, num_sentences=num_sentences)

            session["last_summary"] = result["summary"]

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        original=original,
        result=result,
        error=error,
        length=length,
        uploaded_name=uploaded_name,
    )


@app.route("/download/txt")
def download_txt():
    summary = session.get("last_summary")
    if not summary:
        return redirect(url_for("index"))

    return send_file(
        io.BytesIO(summary.encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name="summary.txt",
    )


@app.route("/download/pdf")
def download_pdf():
    summary = session.get("last_summary")
    if not summary:
        return redirect(url_for("index"))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph("TextSummarizer AI — Generated Summary", styles["Title"]),
        Spacer(1, 0.25 * inch),
    ]

    for para in summary.split("\n"):
        if para.strip():
            story.append(Paragraph(html.escape(para), styles["BodyText"]))
            story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="summary.pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
