import streamlit as st
import re
import docx2txt
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Set page title and layout
st.set_page_config(page_title="Data Analyst Resume Analyzer", page_icon="📊", layout="centered")

# Custom CSS for background and design
st.markdown("""
    <style>
    .main {
        background-image: linear-gradient(to right top, #ffffff, #f0f3f8, #e1e8f1, #d1ddeb, #c2d2e5);
        font-family: 'Segoe UI', sans-serif;
        padding: 2rem;
    }
    h1, h2, h3, h4 {
        color: #00264d;
    }
    .stButton > button {
        background-color: #003366;
        color: white;
        border-radius: 8px;
        height: 40px;
        width: 100%;
    }
    .stFileUploader {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# Load skills from file
with open("skills.txt", "r") as f:
    SKILLS = [skill.strip().lower() for skill in f.readlines()]

# Functions
def extract_text_from_pdf(file_path):
    return extract_text(file_path)

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_info(text):
    email = re.findall(r'\S+@\S+', text)
    phone = re.findall(r'\+?\d[\d -]{8,12}\d', text)
    words = [token.text.lower() for token in nlp(text)]
    found_skills = list(set([word for word in words if word in SKILLS]))

    # Extract education
    education_keywords = ["b.tech", "m.tech", "bachelor", "master", "bsc", "msc", "degree", "engineering"]
    education = [line for line in text.split("\n") if any(edu in line.lower() for edu in education_keywords)]

    # Estimate years of experience
    exp_matches = re.findall(r"(\d+)[+]?\s*(?:years|yrs|year)\s*(?:of)?\s*(?:experience)?", text, re.IGNORECASE)
    experience_years = max([int(x) for x in exp_matches], default=0)

    return {
        "email": email[0] if email else "Not found",
        "phone": phone[0] if phone else "Not found",
        "skills": found_skills,
        "education": education[:2],
        "experience": experience_years
    }

def calculate_score(resume_text, jd_text):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0] * 100

# Streamlit App
st.title("📄 Resume Analyzer")
st.write("Upload your resume and see how well it matches the job requirements.")

uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    with open("temp." + file_type, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text
    if file_type == "pdf":
        resume_text = extract_text_from_pdf("temp.pdf")
    else:
        resume_text = extract_text_from_docx("temp.docx")

    # Extract info
    info = extract_info(resume_text)

    st.markdown("---")
    st.subheader("🔍 Extracted Information")
    st.markdown(f"""
    - **📧 Email**: {info['email']}
    - **📞 Phone**: {info['phone']}
    - **💼 Skills Found**: {", ".join(info['skills']) if info['skills'] else "Not Found"}
    - **🎓 Education**: {' | '.join(info['education']) if info['education'] else 'Not found'}
    - **💼 Experience (estimated years)**: {info['experience']}
    """)

    # Job description
    with open("job_description.txt", "r") as jd:
        job_desc = jd.read()

    # Calculate match score
    score = calculate_score(resume_text, job_desc)

    st.markdown("---")
    st.subheader("📈 Matching Score")
    st.progress(min(score / 100, 1.0))
    st.success(f"✅ Resume matches {score:.2f}% with job description.")

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit | Developed by Iden Abhisheik")