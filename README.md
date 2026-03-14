# Image Caption Generator

Small Flask app that captions uploaded images using Salesforce BLIP (transformers).

## Features
- Upload image via web UI
- Server generates caption using `Salesforce/blip-image-captioning-large`
- Result image and caption shown in the browser

## Prerequisites
- Python 3.8+
- pip
- (Optional) GPU + CUDA for faster model inference

## Install
Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install flask torch transformers pillow
```

## Run
Start the app:

```bash
python app.py
```

Open http://localhost:5001 in your browser.

## Folders
- `templates/` — HTML template used by Flask (`index.html`).
- `uploads/` — temporary uploads (files are removed after processing by default).


## Troubleshooting
- If port 5001 is in use, kill the process or change the port in `app.py`:

```bash
lsof -iTCP:5001 -sTCP:LISTEN -n -P
kill <PID>
# or
sudo kill -9 <PID>
```

- If the ML model fails to load, check available memory or CUDA setup. Running on CPU may be slow.
