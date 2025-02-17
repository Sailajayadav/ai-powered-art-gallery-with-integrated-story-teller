import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import os
import base64
import pandas as pd
app = Flask(__name__)

# ---------------------- Helper Functions ----------------------
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
    # Preprocess the uploaded image
    np_array = np.frombuffer(uploaded_image_data, np.uint8)
    uploaded_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    uploaded_image = cv2.resize(uploaded_image, (224, 224))
    uploaded_hist = extract_color_histogram(uploaded_image)

    # Compare against dataset images
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
    """Render the Sign-Up Page."""
    return render_template("3d.html")

@app.route("/home")
def home():
    """Render the Home Page."""
    return render_template("home.html")

@app.route('/second')
def second():
    return render_template('second.html')
@app.route("/index", methods=["GET", "POST"])
def upload_image():
    """Handle image upload and similarity search."""
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file:
            uploaded_image_data = uploaded_file.read()
            
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))

            # Build the path to the images folder
            dataset_image_dir = os.path.join(BASE_DIR, 'static', 'images')
            dataset_image_paths = [
                os.path.join(dataset_image_dir, img)
                for img in os.listdir(dataset_image_dir)
                if os.path.isfile(os.path.join(dataset_image_dir, img))
            ]
            
            # Find the most similar image
            most_similar_image_path, similarity_score = find_most_similar_image(
                uploaded_image_data, dataset_image_paths
            )
            
            # Check similarity threshold
            similarity_threshold = 0.5  # Cosine similarity threshold (0 to 1)
            if similarity_score < similarity_threshold:
                message = f"Oops! No similar image found. Similarity score: {similarity_score:.2f}"
                return jsonify({
                    "below_threshold": True,
                    "similarity_score": float(similarity_score)
                })

            # Convert the matched image to Base64
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

        response = []
        for _, row in results.iterrows():
            response.append({
                "art_name": row['art_name'],
                "artist_name": row['artist_name'],
                "image_url": row['image_url']
            })

        return jsonify(response)
    return render_template("search.html")

if __name__ == "__main__":
    app.run(debug=True)
