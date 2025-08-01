import streamlit as st
import PyPDF2
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Title
st.title("📄 AI Resume Screener (Gemini + TF-IDF Hybrid)")

# Configure Gemini API
with st.sidebar:
    st.subheader("API Configuration")
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Fallback to user input if not in environment (for local development)
    if not api_key:
        api_key = st.text_input("Enter Google API Key", type="password")
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-pro")  # Use latest stable version
            st.success("✅ API connected successfully!")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    else:
        st.warning("Please provide your Google API key to enable AI features")

# Job Description Input
job_description = st.text_area("📝 Paste the Job Description", height=250)

# Upload Multiple Resumes
uploaded_files = st.file_uploader(
    "📤 Upload Resumes (PDFs)", 
    type="pdf", 
    accept_multiple_files=True
)

# Scoring Method Selection
scoring_method = st.radio(
    "Select Scoring Method:",
    ("TF-IDF (Traditional)", "Gemini AI (Advanced)", "Hybrid Approach"),
    index=2
)

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        return text
    except Exception as e:
        st.error(f"Error reading {file.name}: {str(e)}")
        return ""

# Function to score resume vs job description using TF-IDF
def score_resume_tfidf(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity * 100  # Percentage

# Function to score resume using Gemini AI
def score_resume_gemini(resume_text, jd_text):
    try:
        prompt = f"""
        Analyze this resume against the following job description and provide a score from 0-100.
        Consider factors like relevant skills, experience, education, and overall fit.

        Job Description:
        {jd_text}

        Resume:
        {resume_text}

        Provide only the numerical score (0-100) as your response, no other text.
        """
        response = model.generate_content(prompt)
        return float(response.text)
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        return 0

# Function for hybrid scoring
def score_resume_hybrid(resume_text, jd_text):
    tfidf_score = score_resume_tfidf(resume_text, jd_text)
    gemini_score = score_resume_gemini(resume_text, jd_text)
    # Weighted average (adjust weights as needed)
    return (tfidf_score * 0.3) + (gemini_score * 0.7)

# Processing resumes when both inputs are provided
if uploaded_files and job_description.strip() and api_key:
    st.subheader("🔍 Screening Results")
    qualified = []
    processing_method = {
        "TF-IDF (Traditional)": score_resume_tfidf,
        "Gemini AI (Advanced)": score_resume_gemini,
        "Hybrid Approach": score_resume_hybrid
    }[scoring_method]
    
    with st.spinner("⏳ Analyzing resumes..."):
        for i, file in enumerate(uploaded_files):
            resume_text = extract_text_from_pdf(file)
            if not resume_text.strip():
                st.warning(f"{file.name} - ❌ Could not extract text.")
                continue

            score = processing_method(resume_text, job_description)
            if score >= 80:
                qualified.append((file.name, score))
            
            # Show progress and immediate results
            with st.expander(f"{file.name} - {score:.1f}%"):
                st.write(f"**Score:** {score:.1f}%")
                if scoring_method != "TF-IDF (Traditional)":
                    with st.spinner("Generating AI analysis..."):
                        analysis_prompt = f"""
                        Provide a concise 3-bullet point analysis of how this resume matches the job description,
                        highlighting strengths and potential gaps.

                        Job Description:
                        {job_description}

                        Resume:
                        {resume_text}
                        """
                        analysis = model.generate_content(analysis_prompt)
                        st.write(analysis.text)

    if qualified:
        st.success(f"\n✅ {len(qualified)} resumes matched with score ≥ 80%:\n")
        sorted_results = sorted(qualified, key=lambda x: x[1], reverse=True)
        for name, score in sorted_results:
            st.markdown(f"- 🟢 **{name}** → `{score:.2f}%` match")
    else:
        st.info("⚠️ No resumes scored 80% or more.")
elif uploaded_files and not job_description.strip():
    st.info("ℹ️ Please enter a job description to start screening.")
elif uploaded_files and not api_key:
    st.warning("🔒 Please enter your Gemini API key in the sidebar to enable screening.")