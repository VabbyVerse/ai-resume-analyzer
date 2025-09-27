import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import parsher
import matcher

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_data
def load_and_process_data(data_path, _model):
    try:
        jobs_df = pd.read_csv(data_path)
        if jobs_df.empty:
            return None
        jobs_df['embedding'] = jobs_df['full_description'].apply(lambda x: _model.encode(x) if isinstance(x, str) else np.zeros(384))
        return jobs_df
    except FileNotFoundError:
        return None

jobs_df = load_and_process_data('my_jobs.csv', model)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Upload Your Resume")
    uploaded_resume = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.markdown("---")

    # --- NEW FEATURE: JOB TYPE FILTER ---
    st.header("🔍 Job Filter")
    job_filter = st.selectbox(
        "Select job type",
        ("All Jobs", "Internships Only")
    )
    # --- END OF NEW FEATURE ---
    
    st.markdown("---")
    st.header("About")
    st.info("This AI-powered tool analyzes your resume against a dataset of job descriptions to find your best match.")

# --- MAIN PAGE ---
st.title("🚀 AI Resume Analyzer & Job Matcher")
st.write("Welcome! Upload your resume and select a job type on the left to get started.")

if uploaded_resume:
    if jobs_df is not None:
        
        # --- NEW FEATURE: FILTERING LOGIC ---
        filtered_jobs_df = jobs_df.copy()
        if job_filter == "Internships Only":
            # Filter the DataFrame to only include rows where job_type is 'Internship'
            filtered_jobs_df = jobs_df[jobs_df['job_type'].str.contains("Internship", case=False, na=False)]
        # --- END OF NEW FEATURE ---

        with st.spinner("Analyzing your resume..."):
            resume_data = parsher.parse_resume(uploaded_resume)
        
        if resume_data:
            st.success("Resume analysis complete!")
            
            # ... (Resume Analysis display code remains the same) ...
            st.header("📄 Your Resume Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Your Details")
                st.write(f"**Name:** {resume_data['name']}")
                st.write(f"**Email:** {resume_data['email']}")
                st.write(f"**Experience:** {resume_data['experience']} Years")
            with col2:
                st.subheader("Your Top Skills")
                st.info(", ".join(sorted(resume_data['skills'])[:10]))

            st.markdown("---")

            with st.spinner("Searching for the best job matches..."):
                resume_embedding = matcher.get_embedding(resume_data['text'], model)
                # IMPORTANT: Use the filtered_jobs_df for matching
                top_matches = matcher.find_best_matches(resume_embedding, filtered_jobs_df)

            st.header(f"🎯 Top Job Matches for You ({job_filter})")
            if top_matches:
                # ... (The rest of the display loop for matches remains the same) ...
                for i, match in enumerate(top_matches):
                    with st.expander(f"**{i+1}. {match['job_title']} at {match['company']}** (Match Score: {match['score']:.2%})"):
                        exp_required = match['min_experience_years']
                        exp_resume = resume_data['experience']
                        if exp_resume >= exp_required:
                            st.markdown(f"**Experience:** ✅ Your **{exp_resume}** years meet the required **{exp_required}** years.")
                        else:
                            st.markdown(f"**Experience:** ⚠️ Your **{exp_resume}** years is less than the required **{exp_required}** years.")
                        st.markdown("---")
                        st.subheader("Skill Analysis")
                        job_skills = set(str(match['required_skills']).lower().split(','))
                        resume_skills = set(resume_data['skills'])
                        matched_skills = job_skills.intersection(resume_skills)
                        missing_skills = job_skills - resume_skills
                        if matched_skills:
                            st.success(f"**✅ Matching Skills:** {', '.join(sorted(matched_skills))}")
                        if missing_skills:
                            st.warning(f"**❗️ Missing Skills:** {', '.join(sorted(missing_skills))}")
                        st.markdown("---")
                        st.subheader("Full Job Description")
                        st.write(match['full_description'])
            else:
                st.error(f"Could not find any suitable {job_filter} in the dataset.")
        else:
            st.error("Failed to parse the uploaded resume.")
else:
    st.info("Please upload a resume in the sidebar to begin the analysis.")