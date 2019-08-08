"""A very simple Flask application"""

from os import path
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from .processors.topicgenerator import TopicGenerator
from werkzeug.utils import secure_filename

METHODS = ["GET", "POST"]
UPLOAD_FOLDER = path.join(path.abspath(path.dirname(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"txt"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


def is_valid_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def handle_upload(request):
    if "file" not in request.files:
        flash("No file in request")
        return request.url

    file = request.files["file"]
    if not file.filename:
        flash("No selected file")
        return request.url

    if file and is_valid_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(path.join(app.config["UPLOAD_FOLDER"], filename))
        return url_for("uploaded_file", filename=filename)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        url = handle_upload(request)
        return redirect(url)
    else:
        return render_template("upload.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    tg = TopicGenerator([filename])
    tg.generate_gensim_topics()
    return render_template("uploaded.html", tg=tg)
