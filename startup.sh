#!/bin/bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet
gunicorn -w 2 -k uvicorn.workers.UvicornWorker api.main:app
