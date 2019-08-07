from flask import Flask, escape, url_for, render_template request

app = Flask(__name__)

METHODS = ["GET", "POST"]

@app.route("/")
def index():
    return "index"

@app.route("/upload", methods=METHODS)
def upload_file():
    pass
