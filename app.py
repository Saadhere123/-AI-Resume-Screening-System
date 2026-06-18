import gradio as gr
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def analyze(job_description, files):

    if files is None:
        return "No files uploaded"

    resume_texts = []
    resume_names = []

    for file in files:
        resume_texts.append(extract_text(file))
        resume_names.append(file.name)

    docs = [job_description] + resume_texts

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(docs)

    similarity = cosine_similarity(vectors[0:1], vectors[1:])[0]

    results = []
    for i, score in enumerate(similarity):
        results.append([
            resume_names[i],
            round(score * 100, 2)
        ])

    df = pd.DataFrame(results, columns=["Resume", "Match %"])
    df = df.sort_values(by="Match %", ascending=False)

    top = f"🏆 Top Candidate: {df.iloc[0]['Resume']} ({df.iloc[0]['Match %']}%)"

    return df, top


with gr.Blocks(title="AI Resume Analyzer 🤖") as app:

    gr.Markdown("# 🤖 AI Resume Screening System")

    job = gr.Textbox(label="Job Description")

    files = gr.File(
        label="Upload Resumes (PDF)",
        file_count="multiple",
        file_types=[".pdf"]
    )

    btn = gr.Button("Analyze")

    table = gr.Dataframe(label="Ranking Results")
    output = gr.Textbox(label="Top Candidate")

    btn.click(
        analyze,
        inputs=[job, files],
        outputs=[table, output]
    )

app.launch()
