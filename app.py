from flask import Flask, request, render_template, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and processor
processor = None
model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the BLIP model and processor"""
    global processor, model
    try:
        logger.info("Loading BLIP model...")
        
        # Load processor
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        if processor is None:
            raise Exception("Failed to load processor")
        logger.info("Processor loaded successfully")
        
        # Load model
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        if model is None:
            raise Exception("Failed to load model")
        logger.info("Model loaded successfully")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # Verify model is properly loaded
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Processor type: {type(processor)}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Set globals to None on failure
        processor = None
        model = None
        raise

def generate_caption(image_path):
    """Generate caption for an image"""
    global model, processor
    
    try:
        # Check if model and processor are loaded
        if model is None or processor is None:
            logger.error("Model or processor is None")
            load_model()  # Try to reload
            if model is None or processor is None:
                return "Error: Model not loaded properly"
        
        # Check if image file exists
        if not os.path.exists(image_path):
            return "Error: Image file not found"
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Image loaded successfully: {image.size}")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return f"Error loading image: {str(e)}"
        
        # Process image with error handling
        try:
            inputs = processor(image, return_tensors="pt")
            logger.info("Image processed by processor")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return f"Error processing image: {str(e)}"
        
        # Move inputs to same device as model
        try:
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.info(f"Inputs moved to device: {device}")
        except Exception as e:
            logger.error(f"Error moving inputs to device: {str(e)}")
            return f"Error moving inputs to device: {str(e)}"
        
        # Generate caption
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50, num_beams=5)
            logger.info("Caption generated successfully")
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return f"Error during generation: {str(e)}"
        
        # Decode caption
        try:
            caption = processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Caption decoded: {caption}")
            return caption if caption else "No caption generated"
        except Exception as e:
            logger.error(f"Error decoding caption: {str(e)}")
            return f"Error decoding caption: {str(e)}"
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_caption: {str(e)}")
        return f"Unexpected error: {str(e)}"

def image_to_base64(image_path):
    """Convert image to base64 for display in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Get image format
            with Image.open(image_path) as img:
                img_format = img.format.lower()
                
            return f"data:image/{img_format};base64,{img_base64}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and generate caption"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Generate caption
            caption = generate_caption(filepath)
            
            # Convert image to base64 for display
            image_base64 = image_to_base64(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'caption': caption,
                'image': image_base64,
                'success': True
            })
            
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'processor_loaded': processor is not None,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    })


# Flask will render it with `render_template('index.html')`.
if __name__ == '__main__':
    # Load model on startup
    try:
        load_model()
        print("Starting Flask application...")
        print("Model loaded successfully!")
        print("Open http://localhost:5001 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        print("Make sure you have the required dependencies installed:")
        print("pip install flask torch transformers pillow")