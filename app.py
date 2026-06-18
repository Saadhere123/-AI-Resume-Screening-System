import gradio as gr
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PDF TEXT + PREVIEW
# =========================
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# =========================
# MAIN ANALYSIS FUNCTION
# =========================
def analyze(job_description, files):

    resume_texts = []
    resume_names = []
    previews = []

    for file in files:
        # text extraction
        text = extract_text(file)
        resume_texts.append(text)
        resume_names.append(file.name)

        # PDF preview (Gradio supports direct file display)
        previews.append(file.name)

    documents = [job_description] + resume_texts

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(documents)

    similarity = cosine_similarity(vectors[0:1], vectors[1:])[0]

    results = []
    for i, score in enumerate(similarity):
        results.append([resume_names[i], round(score * 100, 2)])

    df = pd.DataFrame(results, columns=["Resume", "Match %"])
    df = df.sort_values(by="Match %", ascending=False)

    return df, previews


# =========================
# UI DESIGN
# =========================
with gr.Blocks(title="AI Resume Screening System") as app:

    gr.Markdown("""
    # 🤖 AI Resume Screening System  
    Upload resumes, preview PDFs, and get AI-based ranking
    """)

    with gr.Row():
        job_description = gr.Textbox(
            label="Job Description",
            placeholder="Enter job description here..."
        )

    with gr.Row():
        files = gr.File(
            label="Upload Resumes (PDF)",
            file_count="multiple",
            file_types=[".pdf"]
        )

    submit = gr.Button("🚀 Analyze Resumes")

    with gr.Row():
        output_table = gr.Dataframe(label="Ranking Results")
        pdf_preview = gr.File(label="📄 Uploaded Resume Preview")

    submit.click(
        fn=analyze,
        inputs=[job_description, files],
        outputs=[output_table, pdf_preview]
    )

# =========================
# LAUNCH APP
# =========================
app.launch()
