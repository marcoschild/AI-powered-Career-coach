from flask import Flask, request, jsonify
import random
import spacy
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Sample career dataset
career_data = {
    "Software Engineer": ["Python", "Java", "C++", "Machine Learning", "Algorithms"],
    "Data Scientist": ["Python", "SQL", "Statistics", "Deep Learning", "Data Visualization"],
    "Marketing Manager": ["SEO", "Google Ads", "Branding", "Social Media"],
    "UX/UI Designer": ["Adobe XD", "Figma", "User Research", "Prototyping"]
}

# Train a simple ML model for career recommendation
vectorizer = TfidfVectorizer()
career_labels = list(career_data.keys())
career_skills = [" ".join(skills) for skills in career_data.values()]
X = vectorizer.fit_transform(career_skills)
model = NearestNeighbors(n_neighbors=1).fit(X)

def extract_skills(resume_text):
    doc = nlp(resume_text)
    skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return skills

def predict_career(resume_text):
    skills = " ".join(extract_skills(resume_text))
    vectorized_input = vectorizer.transform([skills])
    index = model.kneighbors(vectorized_input, return_distance=False)[0][0]
    return career_labels[index]

def skill_gap_analysis(resume_text, career):
    user_skills = set(extract_skills(resume_text))
    required_skills = set(career_data.get(career, []))
    missing_skills = required_skills - user_skills
    return list(missing_skills)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    resume_text = data.get("resume", "")
    career = predict_career(resume_text)
    missing_skills = skill_gap_analysis(resume_text, career)
    return jsonify({"career": career, "missing_skills": missing_skills})

if __name__ == "__main__":
    app.run(debug=True)
