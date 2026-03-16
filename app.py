from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
import logging
import requests
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('APP_SECRET', 'dev-secret')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB per file

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and processor
processor = None
model = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the BLIP model and processor (local fallback)."""
    global processor, model
    try:
        logger.info("Loading BLIP model locally...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Local model loaded on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        processor = None
        model = None


def generate_caption(image_path):
    """Generate caption for an image."""
    global model, processor

    try:
        if not os.path.exists(image_path):
            return "Error: Image file not found"

        # Try Hugging Face Inference API first
        hf_token = os.environ.get('HUGGINGFACE_API_TOKEN')
        hf_model = os.environ.get('HUGGINGFACE_MODEL', 'Salesforce/blip-image-captioning-base')
        if hf_token:
            try:
                with open(image_path, 'rb') as f:
                    img_bytes = f.read()
                headers = {'Authorization': f'Bearer {hf_token}'}
                url = f'https://api-inference.huggingface.co/models/{hf_model}'
                resp = requests.post(url, headers=headers, data=img_bytes, timeout=30)
                if resp.status_code == 200:
                    ctype = resp.headers.get('Content-Type', '')
                    if 'application/json' not in ctype:
                        logger.warning(f"HF API returned non-JSON content-type: {ctype}")
                    else:
                        data = resp.json()
                        caption = None
                        if isinstance(data, list) and len(data) > 0:
                            first = data[0]
                            if isinstance(first, dict) and 'generated_text' in first:
                                caption = first['generated_text']
                            elif isinstance(first, str):
                                caption = first
                        elif isinstance(data, dict) and 'generated_text' in data:
                            caption = data['generated_text']
                        elif isinstance(data, str):
                            caption = data

                        if isinstance(caption, str) and caption.strip().startswith('<'):
                            logger.warning('HF API returned HTML content as caption')
                            caption = None

                        if caption:
                            logger.info("Caption obtained from Hugging Face Inference API")
                            return caption
                        else:
                            logger.warning("HF API returned no usable caption")
                else:
                    logger.error(f"HF API error {resp.status_code}")
            except Exception as e:
                logger.error(f"Error calling HF Inference API: {e}")

        # Fall back to local model
        if model is None or processor is None:
            logger.warning("Local model not loaded; attempting to load")
            load_model()
            if model is None or processor is None:
                return "Error: No captioning model available"

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return f"Error loading image: {e}"

        try:
            inputs = processor(image, return_tensors="pt")
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return f"Error processing image: {e}"

        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50, num_beams=5)
            caption = processor.decode(out[0], skip_special_tokens=True)
            return caption if caption else "No caption generated"
        except Exception as e:
            logger.error(f"Error during local generation: {e}")
            return f"Error during generation: {e}"

    except Exception as e:
        logger.error(f"Unexpected error in generate_caption: {e}")
        return f"Unexpected error: {e}"


def image_to_base64(image_path):
    """Convert image to base64 for display in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            with Image.open(image_path) as img:
                img_format = img.format.lower()
            return f"data:image/{img_format};base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None


@app.route('/')
def index():
    """Render gallery page"""
    return render_template('gallery.html')


@app.route('/albums', methods=['GET'])
def list_albums():
    """Return album list as JSON"""
    try:
        items = []
        for name in sorted(os.listdir(app.config['UPLOAD_FOLDER'])):
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            if os.path.isdir(path):
                items.append(name)
        return jsonify({'albums': items})
    except Exception as e:
        logger.error(f"Error listing albums: {e}")
        return jsonify({'albums': []})


@app.route('/create_album', methods=['POST'])
def create_album():
    """Create an album and save uploaded images (multi-file)"""
    album_name = request.form.get('album_name') or request.form.get('name')
    if not album_name:
        import time
        album_name = f"album_{int(time.time())}"

    # Sanitize
    album_name = secure_filename(album_name)
    album_path = os.path.join(app.config['UPLOAD_FOLDER'], album_name)
    os.makedirs(album_path, exist_ok=True)

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    saved = []
    for f in files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            filepath = os.path.join(album_path, filename)
            f.save(filepath)
            saved.append(filename)

    return jsonify({'success': True, 'album': album_name, 'files': saved})


@app.route('/album/<album_name>')
def view_album(album_name):
    """Render album view"""
    return render_template('album.html', album_name=album_name)


@app.route('/album/<album_name>/images')
def album_images(album_name):
    """Return images and captions for an album"""
    try:
        album_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(album_name))
        if not os.path.isdir(album_path):
            return jsonify({'error': 'Album not found'}), 404

        items = []
        for fname in sorted(os.listdir(album_path)):
            if allowed_file(fname):
                fpath = os.path.join(album_path, fname)
                caption = generate_caption(fpath)
                img_url = url_for('uploaded_file', filename=f"{album_name}/{fname}")
                items.append({'filename': fname, 'url': img_url, 'caption': caption})

        return jsonify({'images': items})
    except Exception as e:
        logger.error(f"Error fetching album images: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/album/<album_name>/rename', methods=['POST'])
def rename_album(album_name):
    """Rename an album"""
    try:
        data = request.get_json()
        new_name = data.get('new_name', '').strip()
        
        if not new_name:
            return jsonify({'error': 'New name is required'}), 400
        
        # Sanitize names
        old_name = secure_filename(album_name)
        new_name = secure_filename(new_name)
        
        old_path = os.path.join(app.config['UPLOAD_FOLDER'], old_name)
        new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
        
        if not os.path.isdir(old_path):
            return jsonify({'error': 'Album not found'}), 404
        
        if os.path.exists(new_path):
            return jsonify({'error': 'An album with this name already exists'}), 400
        
        # Rename the directory
        os.rename(old_path, new_path)
        
        return jsonify({'success': True, 'new_name': new_name})
    except Exception as e:
        logger.error(f"Error renaming album: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve files from the uploads folder"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle single file upload and generate caption (backwards-compatible)"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            caption = generate_caption(filepath)
            image_base64 = image_to_base64(filepath)
            os.remove(filepath)
            return jsonify({'caption': caption, 'image': image_base64, 'success': True})
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return jsonify({'error': f'Error processing image: {e}'}), 500

    return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'local_model_loaded': model is not None,
        'processor_loaded': processor is not None,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    })


if __name__ == '__main__':
    if os.environ.get('HUGGINGFACE_API_TOKEN'):
        logger.info('Using Hugging Face Inference API for captions')
    else:
        try:
            load_model()
        except Exception as e:
            logger.error(f"Could not load model on startup: {e}")

    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))