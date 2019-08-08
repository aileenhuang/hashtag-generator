python3 -m venv $PWD/venv
source venv/bin/activate
python3 -m pip install -r requirements

export FLASK_APP=eigen-app/app.py
export FLASK_ENV=development

