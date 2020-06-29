python3 -m venv $PWD/venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

export FLASK_APP=hashtags-app/app.py

