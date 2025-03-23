from flask import Flask, request, jsonify, render_template, redirect, url_for
import spacy
import json
import os
import docx2txt
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)

# JSON storage file
JSON_FILE = "data.json"
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to load JSON data
def load_data():
    try:
        with open(JSON_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"job_announcements": [], "cv_records": []}

# Function to save JSON data
def save_data(data):
    with open(JSON_FILE, "w") as file:
        json.dump(data, file, indent=4)

# Preprocessing Functions

def clean_text(text):
    """Normalize text by converting to lowercase and removing unwanted characters."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    text = ' '.join(text.split())  # Normalize whitespace
    return text

def lemmatize_text(text):
    """Lemmatize the text and remove stopwords."""
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def extract_named_entities(text):
    """Extract named entities from text using spaCy's NER."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'GPE', 'PERSON', 'PRODUCT']]

def extract_keywords(text):
    """Extract keywords including lemmatized words, named entities, and TF-IDF keywords."""
    if not text.strip():
        return []

    # Step 1: Clean the text
    cleaned_text = clean_text(text)
    
    # Step 2: Lemmatize the cleaned text
    lemmatized_text = lemmatize_text(cleaned_text)
    
    # Step 3: Extract named entities
    entities = extract_named_entities(lemmatized_text)
    
    # Step 4: Use TF-IDF to extract common keywords
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform([lemmatized_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    keyword_scores = list(zip(feature_names, scores))
    sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
    
    # Step 5: Extract top 10 TF-IDF keywords
    tfidf_keywords = [kw[0] for kw in sorted_keywords[:10]]

    # Combine NER entities, lemmatized words, and TF-IDF keywords
    keywords = list(set(entities + tfidf_keywords))
    return keywords

# Function to extract text from uploaded PDF, DOC, or DOCX files and normalize it
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    text = ""

    if ext == ".pdf":
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + " "
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")

    elif ext in [".doc", ".docx"]:
        try:
            text = docx2txt.process(file_path)
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")

    # Normalize text: Remove excessive whitespace, newlines, and special characters
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())  # Removes extra spaces

    if text:
        print(f"Extracted text from {file_path}: {text[:500]}")  # Log first 500 characters
    else:
        print(f"No text extracted from {file_path}")

    return text

# Function to generate a Boolean search query from extracted keywords
def generate_boolean_query(keywords):
    return "(" + " OR ".join(keywords) + ")" if keywords else "No valid search terms found"

# Function to classify job description
def classify_job(job_description):
    data = load_data()
    
    # Step 1: Preprocess job description (cleaning, lemmatizing, extracting keywords)
    job_keywords = extract_keywords(job_description)
    
    job_entry = {
        "id": f"job_{len(data['job_announcements']) + 1}",
        "description": job_description,
        "keywords": job_keywords,
        "boolean_query": generate_boolean_query(job_keywords),
        "matches": []  # Stores matching CVs
    }
    data["job_announcements"].append(job_entry)
    save_data(data)
    return job_entry

# Function to calculate Precision, Recall, and F1 Score
def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return precision, recall, f1

# Function to match CV with job postings
def match_cv(cv_text):
    data = load_data()

    # Step 1: Preprocess the CV text (cleaning, lemmatizing, extracting keywords)
    cv_keywords = extract_keywords(cv_text)

    true_labels = []  # Ground truth labels for the jobs (1 for match, 0 for no match)
    predicted_labels = []  # Predicted labels based on similarity score

    cv_entry = {
        "id": f"cv_{len(data['cv_records']) + 1}",
        "description": cv_text,
        "keywords": cv_keywords,
    }
    data["cv_records"].append(cv_entry)

    # List to hold the similarity scores
    similarity_scores = []

    for job in data["job_announcements"]:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([" ".join(job["keywords"]), " ".join(cv_keywords)])
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0] * 100

        print(f"Cosine Similarity Score between Job '{job['id']}' and CV '{cv_entry['id']}': {score}%")

        similarity_scores.append({
            "job_id": job['id'],
            "score": score,
            "job_description": job["description"]
        })

        if score >= 50:  # Matching threshold (50% similarity)
            if "matches" not in job:
                job["matches"] = []
            job["matches"].append({
                "cv_id": cv_entry["id"],
                "cv_description": cv_entry["description"],
                "score": score
            })
            true_labels.append(1)  # True positive: there is a match
            predicted_labels.append(1)  # Predicted match
        else:
            true_labels.append(0)  # False negative: no match
            predicted_labels.append(0)  # Predicted no match

    # Sort the similarity scores in descending order
    highest_match = max(similarity_scores, key=lambda x: x['score'])

    # Calculate Precision, Recall, and F1 Score
    precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)

    # Add the metrics to the CV entry
    cv_entry["precision"] = precision
    cv_entry["recall"] = recall
    cv_entry["f1_score"] = f1

    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(f"Highest matching job for CV '{cv_entry['id']}': Job '{highest_match['job_id']}' with score {highest_match['score']}%")

    save_data(data)
    return cv_entry, highest_match  # Return the highest match


# Flask Routes

@app.route("/")
def index():
    """Landing page with navigation links."""
    return render_template("index.html")

@app.route("/job_input", methods=["GET", "POST"])
def job_input():
    if request.method == "POST":
        job_description = request.form.get("job_description", "")
        classify_job(job_description)
        return redirect(url_for("job_list"))
    return render_template("job_input.html")

@app.route("/job_list")
def job_list():
    data = load_data()
    return render_template("job_list.html", jobs=data["job_announcements"], cvs=data["cv_records"])

@app.route("/cv_details/<cv_id>")
def cv_details(cv_id):
    """Displays details of a specific CV."""
    data = load_data()
    cv_entry = next((cv for cv in data["cv_records"] if cv["id"] == cv_id), None)
    if not cv_entry:
        return "CV not found", 404
    return render_template("cv_details.html", cv=cv_entry)

@app.route("/cv_input", methods=["GET", "POST"])
def cv_input():
    if request.method == "POST":
        cv_text = request.form.get("cv_text", "")
        uploaded_file = request.files.get("cv_file")

        if uploaded_file and uploaded_file.filename:
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            extracted_text = extract_text_from_file(file_path)

            if extracted_text:
                cv_text = extracted_text

        if cv_text.strip():
            print(f"Processing CV: {cv_text[:500]}")  # Log first 500 characters
            match_cv(cv_text)
            return redirect(url_for("job_list"))

    return render_template("cv_input.html")

if __name__ == "__main__":
    app.run(debug=True)  