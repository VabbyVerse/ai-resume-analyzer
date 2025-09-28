import re
import spacy
from spacy.matcher import PhraseMatcher
import PyPDF2
from datetime import datetime
import pandas as pd

# IMPORTANT: We are now loading the medium spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading 'en_core_web_md' model... This may take a moment.")
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

def extract_text_from_pdf(file_obj):
    try:
        reader = PyPDF2.PdfReader(file_obj)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text
    except Exception as e:
        return None

def extract_name(text):
    lines = text.split('\n')
    for line in lines[:5]:
        if len(line.strip().split()) in [2, 3]:
            return line.strip()
    return "Name not found"

def extract_email(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else "Email not found"

def extract_skills(text):
    try:
        with open('skills_list.txt', 'r') as f:
            skills_list = [line.strip().lower() for line in f]
    except FileNotFoundError:
        return ["SKILLS FILE NOT FOUND"]
    
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(skill) for skill in skills_list]
    matcher.add("SKILL", patterns)
    
    doc = nlp(text)
    matches = matcher(doc)
    
    found_skills = set(doc[start:end].text.lower() for _, start, end in matches)
    return list(found_skills)

def extract_experience(text):
    date_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+\d{4}|(?:\d{1,2}\/\d{4}))\s*-\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+\d{4}|(?:\d{1,2}\/\d{4})|Present|Current)'
    
    matches = re.finditer(date_pattern, text, re.IGNORECASE)
    total_months = 0
    
    for match in matches:
        start_date = pd.to_datetime(match.group(1), errors='coerce')
        
        if match.group(2).lower() in ['present', 'current']:
            end_date = pd.to_datetime(datetime.now())
        else:
            end_date = pd.to_datetime(match.group(2), errors='coerce')
        
        if pd.notna(start_date) and pd.notna(end_date):
            duration_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            total_months += duration_months

    return round(total_months / 12, 1)

def parse_resume(file_obj):
    text = extract_text_from_pdf(file_obj)
    if not text:
        return None
    
    name = extract_name(text)
    email = extract_email(text)
    skills = extract_skills(text)
    total_experience = extract_experience(text)
    
    return {
        'name': name,
        'email': email,
        'text': text,
        'skills': skills,
        'experience': total_experience
    }