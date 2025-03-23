Automated Job-CV Matching System
Description
This project is an Automated Job-CV Matching System built with Flask and Natural Language Processing (NLP) techniques. The system aims to assist recruiters by automatically matching job descriptions with CVs, streamlining the hiring process.

By extracting relevant keywords from both job descriptions and CVs, the system calculates the similarity between them and provides a match score. If the match score is 50% or higher, the CV is considered relevant for the job description.

Features
Job Description Submission: Users can input job descriptions manually or paste them from external documents.

CV Submission: Users can submit CVs in PDF or DOCX formats, or input them directly.

Keyword Extraction: The system extracts keywords using NLP techniques (e.g., lemmatization, Named Entity Recognition).

Job-CV Matching: The system calculates a match score based on keyword similarity using cosine similarity.

Job-CV Match List: Displays all job announcements with matching CVs and their match scores.

Installation
Requirements
Python 3.7+
Flask
spaCy
scikit-learn
PyPDF2
docx2txt

Steps to Install

Create a virtual environment:

py -3 -m venv venv  
venv\Scripts\Activate.ps1    

Install the required dependencies:

pip install -r requirements.txt

Download the spaCy model:

py -m spacy download en_core_web_sm

Run the Flask app:

py app.py

Open your browser and go to http://127.0.0.1:5000/ to access the app.

File Structure
app.py: Main Flask application file.

templates/: Contains HTML templates used by Flask for rendering.

index.html: The landing page of the app.

job_input.html: Page for submitting job descriptions.

job_list.html: Page to display all job announcements.

cv_input.html: Page for submitting CVs.

cv_details.html: Page to view detailed information about a CV.

uploads/: Directory to store uploaded CV files.

data.json: JSON file to store job descriptions and CV records.

requirements.txt: Lists all the required Python libraries for the project.

Usage
Submit a Job: Go to the "Submit Your Job Announcement" page, input the job description, and submit it. The system will process the job and create a job listing.

Submit a CV: Go to the "Submit Your CV" page, upload a CV file or enter the text manually. The system will extract keywords and compare the CV with job descriptions.

View Matches: Once a CV is submitted, the system will display all matching job announcements along with match percentages.

Technologies Used
Flask: Web framework for building the application.

spaCy: NLP library used for text preprocessing, lemmatization, and Named Entity Recognition.

scikit-learn: Library for computing cosine similarity and TF-IDF vectorization.

PyPDF2: Library for extracting text from PDF files.

docx2txt: Library for extracting text from DOCX files.
