"""A very simple Flask application"""

import simplejson as json
from os import path
from flask import Flask, flash, request, redirect, url_for, render_template
from .processors.topicgenerator import TopicGenerator
from werkzeug.utils import secure_filename

METHODS = ["GET", "POST"]
UPLOAD_FOLDER = path.join(path.abspath(path.dirname(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"txt"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


def _is_valid_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _handle_upload(request):
    if not request.files:
        flash("No file in request")
        return request.url

    file_dict = request.files
    for file_key, file_data in file_dict.items():
        if file_key and file_data.filename:
            if not _is_valid_file(file_data.filename):
                flash("Invalid file submitted")
                return request.url

    filenames_dict = {}  # to be serialized as json and passed in as an URL query param
    for file_key, file_data in file_dict.items():
        if file_data.filename:
            if "filenames" not in filenames_dict:
                filenames_dict["filenames"] = []
            filename = secure_filename(file_data.filename)
            filenames_dict["filenames"].append(filename)
            file_data.save(path.join(app.config["UPLOAD_FOLDER"], filename))
    filenames_json = json.dumps(filenames_dict)

    return url_for("uploaded_file", filenames_json=filenames_json)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        url = _handle_upload(request)
        return redirect(url)
    else:
        return render_template("upload.html")


@app.route("/uploads/<filenames_json>")
def uploaded_file(filenames_json):
    filenames_dict = json.loads(filenames_json)  # deserialize into dict
    tg = TopicGenerator(filenames_dict["filenames"])
    tg.generate_output()

    return render_template("uploaded.html", tg=tg)
