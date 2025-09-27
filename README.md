# 🚀 AI Resume Analyzer & Job Matcher

A Streamlit web application that uses a Sentence-Transformer model to analyze resumes, match them against a custom dataset of job descriptions, and identify skill gaps.

## ✨ Features

- **Resume Parsing:** Extracts text, skills, and work experience from PDF resumes.
- **AI-Powered Matching:** Calculates a semantic similarity score between the resume and job descriptions.
- **Skill Gap Analysis:** Identifies skills required by the job but missing from the resume.
- **Internship Filter:** Allows users to filter for internship opportunities.
- **Interactive UI:** Built with Streamlit for a clean and user-friendly experience.

## 📸 Screenshot

![App Screenshot]("C:\Users\Vaibhav Rathore\OneDrive\Pictures\Screenshots\Screenshot 2025-09-27 201433.png")

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Streamlit, Pandas, Scikit-learn, Sentence-Transformers, spaCy
- **NLP Model:** `all-MiniLM-L6-v2`

## ⚙️ How to Run

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the spaCy model: `python -m spacy download en_core_web_lg`
4. Run the app: `streamlit run app.py`