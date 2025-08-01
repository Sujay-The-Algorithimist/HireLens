# ⭐ HireLens AI – AI-Powered Resume Screening System (Gemini + TF-IDF + Hybrid)

**HireLens** is a powerful, intelligent resume screening tool that blends traditional NLP with Generative AI to evaluate multiple resumes against a job description. Built for recruiters, hiring teams, and developers exploring AI-based ATS systems.

---

## 🚀 Features

- Upload **1 to 20+ PDF resumes** in bulk
- Upload a single **Job Description (JD)** in PDF format
- Uses a **hybrid model** combining:
  - ✨ TF-IDF (Term Frequency–Inverse Document Frequency)
  - 🔍 Cosine Similarity
  - 🤖 Google Gemini (Generative AI via your own API key)
- Returns a **similarity score for each resume**
- Automatically **accepts/rejects** resumes based on a threshold (default: 85%)
- Clean, interactive **Streamlit interface**

---

## 📌 How It Works

The core logic is a **hybrid model**:

1. **TF-IDF + Cosine Similarity (weight: 0.3)**  
   Calculates how well each resume matches the job description based on traditional NLP vector space modeling.

2. **Gemini Score (weight: 0.7)**  
   Uses Google Gemini (via API key) to semantically compare resumes with the job description for deeper, context-aware analysis.

3. **Final Score = 0.7 × Gemini Score  + 0.3 × TF-IDF Score**

If the final score is **≥ 85**, the resume is **accepted**. Else, **rejected**.

---

## 🧠 Tech Stack

- Python 🐍
- Streamlit
- PyPDF2
- scikit-learn (TF-IDF + Cosine Similarity)
- Google Generative AI (Gemini API)

---

## 📸 Screenshots

*(Insert screenshots of the UI, results table, score chart here)*

---

## 🔧 Setup Instructions

1. **Clone the Repo**
   ```bash
   git clone https://github.com/yourusername/hirelens.git
   cd hirelens


## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smarthire-ai.git
cd smarthire-ai
