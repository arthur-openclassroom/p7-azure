#!/bin/bash
set -e
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet
gunicorn -w 2 -k uvicorn.workers.UvicornWorker --bind "0.0.0.0:${PORT:-8000}" api.main:app
