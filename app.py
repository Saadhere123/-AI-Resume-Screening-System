import streamlit as st
import pdfplumber
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)

# =========================
# CUSTOM UI DESIGN
# =========================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg,#0f172a,#1e293b,#312e81);
}

.hero {
    padding:30px;
    border-radius:20px;
    text-align:center;
    color:white;
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    margin-bottom:20px;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.1);
    border-radius:15px;
    padding:20px;
    backdrop-filter: blur(10px);
    color:white;
}

.stButton>button {
    background:#2563eb;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:18px;
}

.stButton>button:hover {
    background:#1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION
# =========================
st.markdown("""
<div class="hero">
<h1>🤖 AI Resume Screening System</h1>
<h4>Smart Candidate Ranking Using Machine Learning</h4>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("📊 Dashboard")
    st.info("""
    ✔ Upload Resumes  
    ✔ Enter Job Description  
    ✔ AI Analysis  
    ✔ Ranking System  
    """)

# =========================
# INPUTS
# =========================
job_description = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# =========================
# ANALYSIS
# =========================
if st.button("🚀 Analyze Resumes"):

    if uploaded_files and job_description:

        with st.spinner("Analyzing Resumes with AI..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

        resume_texts = []
        resume_names = []

        for file in uploaded_files:
            resume_texts.append(extract_text(file))
            resume_names.append(file.name)

        documents = [job_description] + resume_texts

        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform(documents)

        similarity_scores = cosine_similarity(
            vectors[0:1],
            vectors[1:]
        )

        results = []

        for i, score in enumerate(similarity_scores[0]):
            results.append({
                "Resume": resume_names[i],
                "Match %": round(score * 100, 2)
            })

        df = pd.DataFrame(results)
        df = df.sort_values(by="Match %", ascending=False)

        df["Rank"] = range(1, len(df) + 1)

        best_candidate = df.iloc[0]

        # =========================
        # METRICS
        # =========================
        col1, col2, col3 = st.columns(3)

        col1.metric("📄 Total Resumes", len(df))
        col2.metric("🏆 Best Match", f"{df['Match %'].max()}%")
        col3.metric("📊 Average Score", f"{round(df['Match %'].mean(),2)}%")

        # =========================
        # ATS BADGE
        # =========================
        score = best_candidate["Match %"]

        if score >= 80:
            st.success("⭐⭐⭐⭐⭐ Excellent ATS Match")
        elif score >= 60:
            st.warning("⭐⭐⭐⭐ Good ATS Match")
        else:
            st.error("⭐⭐ Needs Improvement")

        # =========================
        # TOP CANDIDATE CARD
        # =========================
        st.markdown(f"""
        <div style="
        background:#10b981;
        padding:20px;
        border-radius:15px;
        color:white;
        text-align:center;
        margin:20px 0;
        ">
        <h2>🏆 Top Candidate</h2>
        <h3>{best_candidate['Resume']}</h3>
        <h1>{best_candidate['Match %']}%</h1>
        </div>
        """, unsafe_allow_html=True)

        # =========================
        # DATAFRAME
        # =========================
        st.dataframe(df, use_container_width=True)

        # =========================
        # BAR CHART
        # =========================
        fig = px.bar(
            df,
            x="Resume",
            y="Match %",
            title="Candidate Ranking"
        )
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # PIE CHART
        # =========================
        fig2 = px.pie(
            df,
            names="Resume",
            values="Match %",
            title="Resume Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # GAUGE CHART
        # =========================
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Best Candidate Score"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig3, use_container_width=True)

        # =========================
        # PROGRESS BARS
        # =========================
        st.subheader("📊 Individual Scores")

        for _, row in df.iterrows():
            st.write(f"{row['Resume']} - {row['Match %']}%")
            st.progress(int(row["Match %"]))

        # =========================
        # DOWNLOAD
        # =========================
        csv = df.to_csv(index=False)

        st.download_button(
            "📥 Download Report",
            csv,
            "resume_analysis.csv",
            "text/csv"
        )