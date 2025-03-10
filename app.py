import os
import unicodedata
from flask import Flask, render_template, request, jsonify, redirect, flash
from werkzeug.utils import secure_filename  # Import secure_filename
import base64
import pandas as pd
import cv2
import numpy as np
#from transformers import pipeline
#from transformers import pipeline, ViTFeatureExtractor, ViTForImageClassification, BlipProcessor, BlipForConditionalGeneration
#import torch
from PIL import Image
import requests
# ... (keep all your existing imports)
import google.generativeai as genai
import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from admin_routes import admin_blueprint



app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key

def get_db_connection():
    conn = sqlite3.connect('virtual_gallery.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create images table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        user_id INTEGER,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Check if admin account exists, if not create it
    cursor.execute("SELECT * FROM users WHERE username = 'admin' AND is_admin = 1")
    admin = cursor.fetchone()
    
    if not admin:
        admin_password = generate_password_hash('admin123')
        cursor.execute(
            "INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)",
            ('admin', 'admin@virtualgallery.com', admin_password, 1)
        )
    
    conn.commit()
    conn.close()

# Call init_db() at startup
with app.app_context():
    init_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'login_error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'is_admin' not in session or not session['is_admin']:
            flash('Admin access required', 'admin_error')
            return redirect(url_for('login', tab='admin'))
        return f(*args, **kwargs)
    return decorated_function

#def zero_shot_classification(text, candidate_labels):
#    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#    result = classifier(text, candidate_labels)
#    return result

'''class ZeroShotImageClassifier:
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
    '''
app.register_blueprint(admin_blueprint)
# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_IMAGES = 8




def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    """Normalize and sanitize filenames to avoid Unicode errors."""
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('utf-8')
    return secure_filename(filename)
# Add Gemini configuration
GOOGLE_API_KEY = 'AIzaSyDQZpVOrAyyyY1qE3-bWNKZPPPHM2zxAxQ'
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

@app.route("/", methods=["GET", "POST"])
def signup():
    """Render the login/sign-up page and handle authentication."""
    if request.method == "POST":
        form_type = request.form.get('form_type', '')
        
        # Handle User Registration
        if form_type == 'register':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            # Validate input
            if not all([username, email, password, confirm_password]):
                flash('All fields are required', 'register_error')
                return redirect(url_for('signup', tab='register'))
                
            if password != confirm_password:
                flash('Passwords do not match', 'register_error')
                return redirect(url_for('signup', tab='register'))
            
            # Check if user already exists
            conn = get_db_connection()
            existing_user = conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', 
                                      (username, email)).fetchone()
            
            if existing_user:
                conn.close()
                flash('Username or email already exists', 'register_error')
                return redirect(url_for('signup', tab='register'))
            
            # Create new user
            hashed_password = generate_password_hash(password)
            conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                      (username, email, hashed_password))
            conn.commit()
            
            # Get the newly created user
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            conn.close()
            
            # Log the user in
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            
            flash('Account created successfully!', 'success')
            return redirect(url_for('home'))
            
        # Handle User Login
        elif form_type == 'login':
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not all([username, password]):
                flash('Username and password are required', 'login_error')
                return redirect(url_for('signup', tab='login'))
            
            conn = get_db_connection()
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            conn.close()
            
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['is_admin'] = user['is_admin']
                
                return redirect(url_for('home'))
            else:
                flash('Invalid username or password', 'login_error')
                return redirect(url_for('signup', tab='login'))
                
        # Handle Admin Login
        elif form_type == 'admin':
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not all([username, password]):
                flash('Username and password are required', 'admin_error')
                return redirect(url_for('signup', tab='admin'))
            
            conn = get_db_connection()
            admin = conn.execute('SELECT * FROM users WHERE username = ? AND is_admin = 1', 
                              (username,)).fetchone()
            conn.close()
            
            if admin and check_password_hash(admin['password'], password):
                session['user_id'] = admin['id']
                session['username'] = admin['username']
                session['is_admin'] = admin['is_admin']
                
                return redirect(url_for('home'))
            else:
                flash('Invalid admin credentials', 'admin_error')
                return redirect(url_for('signup', tab='admin'))
    
    # GET request - render the login/signup page
    return render_template("signup.html")  # Renamed from signup.html to login.html
@app.route('/test')
def test():
    """Render the Test Page."""
    return render_template('index1.html')
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
def upload_images():
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
            
            similarity_threshold = 1 # Cosine similarity threshold
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
    
@app.route('/upload_image', methods=['GET', 'POST'])
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
            
            # Get the most similar image
            most_similar_image_path, similarity_score = find_most_similar_image(
                uploaded_image_data, dataset_image_paths
            )
            
            similarity_threshold = 1  # Adjusted threshold for better matching
            if similarity_score < similarity_threshold:
                return jsonify({
                    "below_threshold": True,
                    "similarity_score": float(similarity_score)
                })

            # Get the image file name from the path
            image_filename = os.path.basename(most_similar_image_path)
            
            # Get artwork data from Excel
            excel_path = os.path.join(BASE_DIR, 'static', 'data', 'artwork.xlsx')
            if not os.path.exists(excel_path):
                return jsonify({"error": "Artworks data not found."}), 404

            # Read the Excel file
            try:
                df = pd.read_excel(excel_path)
                
                # Find matching artwork record based on image filename or URL
                # Assuming the image_url column contains the filename or full path
                # Adjust this logic based on how your Excel data is structured
                matched_artwork = None
                for _, row in df.iterrows():
                    image_url = row['image_url']
                    if image_filename in image_url or image_url in most_similar_image_path:
                        matched_artwork = {
                            "art_name": row['art_name'], 
                            "artist_name": row['artist_name'], 
                            "image_url": row['image_url']
                        }
                        break
                
                # If no match found in Excel, still return the image but with placeholder info
                if not matched_artwork:
                    matched_artwork = {
                        "art_name": "Unknown Artwork", 
                        "artist_name": "Unknown Artist", 
                        "image_url": f"static/images/{image_filename}"
                    }
                
                # Convert the image to base64 for sending to frontend
                similar_image_base64 = image_to_base64(most_similar_image_path)
                
                # Return both the artwork data and the image
                return jsonify({
                    "artwork": matched_artwork,
                    "most_similar_image": f"data:image/jpeg;base64,{similar_image_base64}",
                    "similarity_score": float(similarity_score)
                })
                
            except Exception as e:
                return jsonify({
                    "error": f"Error processing artwork data: {str(e)}"
                }), 500

        return jsonify({"error": "No file uploaded."}), 400
        
    return jsonify({"error": "Method not allowed."}), 405

def find_most_similar_image(uploaded_image_data, dataset_image_paths):
    """Find the most similar image in the dataset to the uploaded image."""
    # Convert uploaded image bytes to OpenCV format
    nparr = np.frombuffer(uploaded_image_data, np.uint8)
    uploaded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Calculate histogram of uploaded image
    uploaded_hist = calculate_image_histogram(uploaded_img)
    
    # Find most similar image
    max_similarity = -1
    most_similar_path = None
    
    for img_path in dataset_image_paths:
        try:
            # Read dataset image
            dataset_img = cv2.imread(img_path)
            if dataset_img is None:
                continue
                
            # Calculate histogram of dataset image
            dataset_hist = calculate_image_histogram(dataset_img)
            
            # Calculate similarity between histograms
            similarity = cv2.compareHist(uploaded_hist, dataset_hist, cv2.HISTCMP_CORREL)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_path = img_path
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return most_similar_path, max_similarity

def calculate_image_histogram(img):
    """Calculate histogram of image for comparison."""
    # Convert to HSV for better color comparison
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], 
                       [0, 256, 0, 256, 0, 256])
    
    # Normalize histogram
    cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    
    return hist

def image_to_base64(image_path):
    """Convert an image to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')





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
    # Define the upload directory to match template expectations
    UPLOAD_DIR = os.path.join(app.root_path, 'static', 'images', 'uploaded_images')
    
    # Ensure the directory exists
       
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
                
                # Create paths with the correct directory structure
                temp_path = os.path.join(UPLOAD_DIR, f'temp_{filename}')
                final_path = os.path.join(UPLOAD_DIR, filename)
                
                # Save temporarily for analysis
                file.save(temp_path)
                
                # Check content with Gemini
                is_safe, message = analyze_image_with_gemini(temp_path)
                
                if is_safe:
                    # Move to final location if safe
                    os.rename(temp_path, final_path)
                    
                    # Store just the filename for template use
                    # This matches the template's logic that extracts filename from path
                    uploaded_images.append(filename)
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


@app.route("/ar_museum")
def ar_museum():
    """Render the AR Museum Page with verified safe images."""
    # Use the absolute path for file operations
    images_dir = os.path.join(app.root_path, 'static', 'images', 'uploaded_images')
    
    # Ensure the directory exists
    os.makedirs(images_dir, exist_ok=True)
    
    safe_images = []
    
    try:
        # Verify all existing images
        for img in os.listdir(images_dir):
            if img.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                img_path = os.path.join(images_dir, img)
                
                try:
                    is_safe, message = analyze_image_with_gemini(img_path)
                    
                    if is_safe:
                        # For template use, we need relative paths from static folder
                        relative_path = os.path.join('images', 'uploaded_images', img)
                        safe_images.append(relative_path)
                    else:
                        # Remove unsafe images
                        os.remove(img_path)
                        app.logger.warning(f"Removed unsafe image during verification: {img} - Reason: {message}")
                except Exception as e:
                    app.logger.error(f"Error analyzing image {img}: {str(e)}")
                    # Skip this image if analysis fails
                    continue
    except FileNotFoundError:
        app.logger.warning("Upload directory not found. Creating directory.")
        os.makedirs(images_dir, exist_ok=True)
    
    return render_template("ar_museum.html", images=safe_images)


if __name__ == "__main__":
    app.run(debug=True)
