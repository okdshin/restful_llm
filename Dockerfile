FROM python:3.9-bullseye

COPY app.py /repo/app.py
COPY templates/index.html /repo/templates/index.html
COPY templates/layout.html /repo/templates/layout.html
COPY static/js/ReplyCard.js /repo/static/js/ReplyCard.js
COPY static/js/UserPromptForm.js /repo/static/js/UserPromptForm.js
COPY requirements.txt /repo/requirements.txt

RUN pip install -r /repo/requirements.txt
