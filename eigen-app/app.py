import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

METHODS = ["GET", "POST"]
UPLOAD_FOLDER = "/uploads"
ALLOWED_EXTENSIONS = {"txt"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def is_valid_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_upload(request):
    # Check that post req has files
    if 'file' not in request.files:
        flash('No file part')
        return request.url

    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if not file.filename:
        flash('No selected file')
        return request.url

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return url_for('uploaded_file', filename=filename)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        url = handle_upload(request)
        redirect(url)
    else:
        return render_template("upload.html")

