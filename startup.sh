#!/bin/bash
set -e
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet
gunicorn --config gunicorn.conf.py api.main:app
