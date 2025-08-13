import streamlit as st
import PyPDF2
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .score-excellent {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .score-good {
        background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .score-poor {
        background: linear-gradient(90deg, #fc4a1a 0%, #f7b733 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    .results-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="main-header">
    <h1>🎯 AI Resume Screener</h1>
    <p>Powered by Gemini AI & Advanced Analytics</p>
</div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # API Key Section
    st.markdown("#### 🔑 API Setup")
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        api_key = st.text_input("Google API Key", type="password", help="Enter your Gemini API key")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            st.success("✅ Connected to Gemini AI")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    else:
        st.warning("⚠️ API key required for AI features")

    st.divider()

    # Scoring Method
    st.markdown("#### 🎯 Scoring Method")
    scoring_method = st.selectbox(
        "Choose analysis approach:",
        ("Hybrid Approach", "Gemini AI (Advanced)", "TF-IDF (Traditional)"),
        help="Hybrid combines both AI and traditional methods for best results"
    )

    st.divider()

    # Threshold Setting
    st.markdown("#### 📊 Qualification Threshold")
    threshold = st.slider("Minimum score for qualification", 60, 95, 80, 5)

    st.divider()

    # Session Stats
    if 'processed_resumes' not in st.session_state:
        st.session_state.processed_resumes = 0
    st.markdown("#### 📈 Session Stats")
    st.metric("Resumes Processed", st.session_state.processed_resumes)

# -------------------- MAIN INPUT AREA --------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Job Description")
    job_description = st.text_area(
        "Paste the complete job description here:",
        height=300,
        placeholder="Enter the job requirements, skills needed, experience level, etc...",
        help="The more detailed the job description, the better the matching accuracy"
    )
    if job_description:
        word_count = len(job_description.split())
        st.caption(f"📊 Word count: {word_count}")

with col2:
    st.markdown("### 📤 Resume Upload")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop your PDF resumes here",
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple PDF resumes for batch processing"
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resume(s) uploaded")
        for file in uploaded_files:
            st.caption(f"📄 {file.name}")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FUNCTIONS --------------------
@st.cache_data
def extract_text_from_pdf(file_bytes, filename):
    try:
        reader = PyPDF2.PdfReader(file_bytes)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        return text.strip()
    except Exception as e:
        st.error(f"Error reading {filename}: {str(e)}")
        return ""

def score_resume_tfidf(resume_text, jd_text):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        vectors = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return float(similarity * 100)
    except:
        return float(0)

def score_resume_gemini(resume_text, jd_text):
    try:
        prompt = f"""
        As an expert HR professional, analyze this resume against the job description.
        Provide a score from 0-100...
        Job Description: {jd_text}
        Resume: {resume_text}
        """
        response = model.generate_content(prompt)
        return float(response.text.strip())
    except Exception as e:
        st.error(f"AI scoring error: {str(e)}")
        return float(0)

def score_resume_hybrid(resume_text, jd_text):
    tfidf_score = float(score_resume_tfidf(resume_text, jd_text))
    gemini_score = float(score_resume_gemini(resume_text, jd_text))
    return float((tfidf_score * 0.3) + (gemini_score * 0.7))

def get_score_style(score):
    if score >= 85:
        return "score-excellent", "🟢"
    elif score >= 70:
        return "score-good", "🟡"
    else:
        return "score-poor", "🔴"

def generate_analysis(resume_text, jd_text, score):
    try:
        prompt = f"""
        Provide a professional analysis of this resume match (Score: {score:.1f}%).
        **Strengths:**
        • [Key strength 1]
        • [Key strength 2]
        **Areas for Improvement:**
        • [Gap 1]
        • [Gap 2]
        **Recommendation:**
        [Brief recommendation]
        Job Description: {jd_text}
        Resume: {resume_text}
        """
        response = model.generate_content(prompt)
        return response.text
    except:
        return "Analysis unavailable - please check API connection."

# -------------------- PROCESSING LOGIC --------------------
if st.button("🚀 Start Resume Screening", type="primary", use_container_width=True):
    if not uploaded_files:
        st.error("❌ Please upload at least one resume")
    elif not job_description.strip():
        st.error("❌ Please enter a job description")
    elif not api_key and scoring_method != "TF-IDF (Traditional)":
        st.error("❌ API key required for AI-powered analysis")
    else:
        st.markdown("---")
        st.markdown("## 🔍 Screening Results")

        processing_method = {
            "TF-IDF (Traditional)": score_resume_tfidf,
            "Gemini AI (Advanced)": score_resume_gemini,
            "Hybrid Approach": score_resume_hybrid
        }[scoring_method]

        qualified_candidates = []
        all_results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            progress_bar.progress((i + 1) / len(uploaded_files))

            resume_text = extract_text_from_pdf(file, file.name)
            if not resume_text:
                st.warning(f"⚠️ Could not extract text from {file.name}")
                continue

            score = float(processing_method(resume_text, job_description))
            all_results.append((file.name, score, resume_text))

            if score >= threshold:
                qualified_candidates.append((file.name, score))

        progress_bar.empty()
        status_text.empty()
        st.session_state.processed_resumes += len(uploaded_files)

        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Resumes", len(uploaded_files))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Qualified Candidates", len(qualified_candidates))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            qualification_rate = (len(qualified_candidates) / len(uploaded_files)) * 100 if uploaded_files else 0
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Qualification Rate", f"{qualification_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Detailed Results
        st.markdown("### 📊 Detailed Analysis")
        sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)

        for name, score, resume_text in sorted_results:
            style_class, emoji = get_score_style(score)
            with st.expander(f"{emoji} {name} - {score:.1f}%", expanded=bool(score >= threshold)):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f'<div class="{style_class}">Score: {score:.1f}%</div>', unsafe_allow_html=True)
                    if score >= threshold:
                        st.success("✅ QUALIFIED")
                    else:
                        st.error("❌ Not Qualified")
                with col2:
                    if scoring_method != "TF-IDF (Traditional)" and api_key:
                        with st.spinner("Generating AI analysis..."):
                            analysis = generate_analysis(resume_text, job_description, score)
                            st.markdown(analysis)
                    else:
                        st.info("💡 Enable AI analysis for detailed insights")

        # Final Summary
        if qualified_candidates:
            st.success(f"🎉 Found {len(qualified_candidates)} qualified candidate(s)!")
            st.markdown("### 🏆 Top Candidates")
            top_candidates = sorted(qualified_candidates, key=lambda x: x[1], reverse=True)
            for i, (name, score) in enumerate(top_candidates[:3], 1):
                st.markdown(f"**{i}.** {name} - **{score:.1f}%** match")
        else:
            st.warning(f"⚠️ No candidates met the {threshold}% threshold. Consider reviewing requirements or lowering the threshold.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 Powered by Google Gemini AI | Built with Streamlit</p>
    <p><small>For best results, ensure job descriptions are detailed and resumes are in good quality PDF format.</small></p>
</div>
""", unsafe_allow_html=True)
