import os
import unicodedata
from flask import Flask, render_template, request, jsonify, redirect, flash
from werkzeug.utils import secure_filename  # Import secure_filename
import base64
import pandas as pd
import cv2
import numpy as np
from transformers import pipeline
from transformers import pipeline, ViTFeatureExtractor, ViTForImageClassification, BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import requests
# ... (keep all your existing imports)
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key

def zero_shot_classification(text, candidate_labels):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(text, candidate_labels)
    return result

class ZeroShotImageClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the zero-shot image classifier using a pre-trained CLIP model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.classifier = pipeline("zero-shot-image-classification", model=self.model, device=0 if torch.cuda.is_available() else -1)

    def load_image(self, image_path):
        """
        Load and preprocess the image.
        """
        if image_path.startswith("http"):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        return image

    def classify_image(self, image_path, candidate_labels):
        """
        Perform zero-shot classification on the image with given candidate labels.
        """
        image = self.load_image(image_path)
        result = self.classifier(image, candidate_labels)
        return result
    
def analyze_image(image_path):
    """
    Analyzes the image and generates a textual description using a pre-trained model.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Use a pre-trained BLIP model for image captioning
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Process the image and generate a caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption
# Configuration
UPLOAD_FOLDER = os.path.join('static', 'images', 'uploaded_images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_IMAGES = 8

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    """Normalize and sanitize filenames to avoid Unicode errors."""
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('utf-8')
    return secure_filename(filename)
# Add Gemini configuration
GOOGLE_API_KEY = 'AIzaSyDgnmM1B99dEkNh07E9JmMl8L1wSwYDg4k'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def preprocess_image(image_path):
    """Read and preprocess an image."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Resize to a fixed size
    return image

def extract_color_histogram(image):
    """Extract color histogram features."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)  # Normalize the histogram
    return hist.flatten()

def find_most_similar_image(uploaded_image_data, dataset_image_paths):
    """Find the most similar image based on color histograms."""
    np_array = np.frombuffer(uploaded_image_data, np.uint8)
    uploaded_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    uploaded_image = cv2.resize(uploaded_image, (224, 224))
    uploaded_hist = extract_color_histogram(uploaded_image)

    similarities = []
    for path in dataset_image_paths:
        dataset_image = preprocess_image(path)
        dataset_hist = extract_color_histogram(dataset_image)

        # Compute similarity (Cosine Similarity)
        similarity = np.dot(uploaded_hist, dataset_hist) / (
            np.linalg.norm(uploaded_hist) * np.linalg.norm(dataset_hist)
        )
        similarities.append(similarity)

    most_similar_index = np.argmax(similarities)
    return dataset_image_paths[most_similar_index], similarities[most_similar_index]

def image_to_base64(image_path):
    """Convert image to Base64 for embedding in HTML."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# ---------------------- Routes ----------------------

@app.route("/")
def signup():
    """Render the Sign-Up Page."""
    return render_template("signup.html")

@app.route("/3d")
def Threed():
    """Render the 3D Page."""
    return render_template("3d.html")

@app.route("/home")
def home():
    """Render the Home Page."""
    return render_template("home.html")

@app.route('/second')
def second():
    """Render the Second Page."""
    return render_template('second.html')

@app.route("/index", methods=["GET", "POST"])
def upload_image():
    """Handle image upload and similarity search."""
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file:
            uploaded_image_data = uploaded_file.read()
            
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            dataset_image_dir = os.path.join(BASE_DIR, 'static', 'images')
            dataset_image_paths = [
                os.path.join(dataset_image_dir, img)
                for img in os.listdir(dataset_image_dir)
                if os.path.isfile(os.path.join(dataset_image_dir, img))
            ]
            
            most_similar_image_path, similarity_score = find_most_similar_image(
                uploaded_image_data, dataset_image_paths
            )
            
            similarity_threshold = 0.5  # Cosine similarity threshold
            if similarity_score < similarity_threshold:
                return jsonify({
                    "below_threshold": True,
                    "similarity_score": float(similarity_score)
                })

            similar_image_base64 = image_to_base64(most_similar_image_path)
            return jsonify({
                "most_similar_image": f"data:image/jpeg;base64,{similar_image_base64}",
                "similarity_score": float(similarity_score)
            })

    return render_template("index.html")

@app.route("/search", methods=["GET", "POST"])
def search():
    """Handle search queries and return matching artworks."""
    if request.method == "POST":
        query = request.form.get("query", "").strip().lower()
        if not query:
            return jsonify({"error": "No search query provided."}), 400

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        excel_path = os.path.join(BASE_DIR, 'static', 'data', 'artwork.xlsx')
        if not os.path.exists(excel_path):
            return jsonify({"error": "Artworks data not found."}), 404

        df = pd.read_excel(excel_path)
        results = df[
            df['art_name'].str.lower().str.contains(query) |
            df['artist_name'].str.lower().str.contains(query)
        ]

        response = [
            {"art_name": row['art_name'], "artist_name": row['artist_name'], "image_url": row['image_url']}
            for _, row in results.iterrows()
        ]

        return jsonify(response)

    return render_template("search.html")




def analyze_image_with_gemini(image_path):
    """
    Analyze image content using Google's Gemini AI
    Returns: (is_safe, message)
    """
    try:
        img = Image.open(image_path)
        
        prompt = """
        Analyze this image for inappropriate or adult content. 
        Consider:
        1. Nudity or explicit content
        2. Violence or gore
        3. Offensive gestures or symbols
        4. Inappropriate text or messages
        5. Check whether the image is AI generated or not
        6. Check properly if the image is AI generated or not
        
        Respond with either:
        SAFE: If the image is appropriate for all audiences
        or
        UNSAFE: If the image contains any inappropriate content or if the image is AI generated or generated by any other software. 
        
        Follow with a brief explanation.
        """

        response = model.generate_content([prompt, img])
        
        # Log the analysis
        


        is_safe = response.text.upper().startswith('SAFE')
        message = response.text.split('\n')[0]

        return is_safe, message

    except Exception as e:
        app.logger.error(f"Error analyzing image {image_path}: {str(e)}")
        return False, f"Error analyzing image: {str(e)}"

# Modify your upload route to include content moderation
@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Handle image uploads with content moderation."""
    if request.method == "POST":
        if 'images' not in request.files:
            flash('No files part')
            return redirect(request.url)

        files = request.files.getlist('images')
        if not files:
            flash('No files selected for uploading')
            return redirect(request.url)
        if len(files) > MAX_IMAGES:
            flash(f'You can upload up to {MAX_IMAGES} images')
            return redirect(request.url)
        
        uploaded_images = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = sanitize_filename(file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{filename}')
                
                # Save temporarily for analysis
                file.save(temp_path)
                
                # Check content with Gemini
                is_safe, message = analyze_image_with_gemini(temp_path)
                
                if is_safe:
                    # Move to final location if safe
                    final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    os.rename(temp_path, final_path)
                    relative_path = os.path.join('static','images', 'uploaded_images', filename).replace(os.sep, '/')
                    uploaded_images.append(relative_path)
                    flash(f'Image {filename} uploaded successfully')
                else:
                    # Remove unsafe image
                    os.remove(temp_path)
                    flash(f'Image {filename} rejected: {message} or the Image is AI generated')
            else:
                flash('Invalid file format')
                return redirect(request.url)
        
        return render_template("upload_results.html", images=uploaded_images)
    
    return render_template("upload_image.html")
# Modify your ar_museum route to include content verification
@app.route("/ar_museum")
def ar_museum():
    """Render the AR Museum Page with verified safe images."""
    images_dir = os.path.join('static', 'images', 'uploaded_images')
    safe_images = []
    
    # Verify all existing images
    for img in os.listdir(images_dir):
        if img.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            img_path = os.path.join(images_dir, img)
            is_safe, _ = analyze_image_with_gemini(img_path)
            
            if is_safe:
                relative_path = os.path.join('static','images', 'uploaded_images', img)
                safe_images.append(relative_path)
            else:
                # Remove unsafe images
                os.remove(img_path)
                app.logger.warning(f"Removed unsafe image during verification: {img}")



    return render_template("ar_museum.html", images=safe_images)


if __name__ == "__main__":
    app.run(debug=True)
