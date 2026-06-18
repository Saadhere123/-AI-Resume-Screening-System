# 🤖 AI Resume Screening System

An AI-powered Resume Screening and Candidate Ranking System built with Streamlit, Machine Learning, and Natural Language Processing (NLP).

The application analyzes uploaded resumes against a given job description and ranks candidates based on their relevance score using TF-IDF Vectorization and Cosine Similarity.

---

## 🚀 Features

* Upload Multiple PDF Resumes
* Enter Custom Job Description
* Resume Text Extraction using PDF Processing
* AI-Based Resume Matching
* Candidate Ranking System
* Match Percentage Calculation
* Interactive Dashboard
* Progress Bar Animation
* Download Results as CSV
* Interactive Charts with Plotly
* Professional User Interface

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-Learn
* Pandas
* PDFPlumber
* Plotly
* Machine Learning
* Natural Language Processing (NLP)

---

## 📂 Project Structure

```text
AI-Resume-Screening-System/
│
├── app.py
├── requirements.txt
├── README.md
└── sample_resumes/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AI-Resume-Screening-System.git
cd AI-Resume-Screening-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 📊 How It Works

1. User enters a Job Description.
2. User uploads one or more PDF resumes.
3. The system extracts text from each resume.
4. TF-IDF converts text into numerical vectors.
5. Cosine Similarity compares resumes with the job description.
6. Candidates are ranked according to similarity scores.
7. Results are displayed in tables and charts.

---

## 🧠 Machine Learning Workflow

```text
Job Description
        │
        ▼
Text Preprocessing
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Cosine Similarity
        │
        ▼
Resume Scoring
        │
        ▼
Candidate Ranking
```

---

## 📈 Future Enhancements

* ATS Score Calculation
* Skill Extraction
* Resume Summarization
* Interview Question Generation
* OpenAI/Gemini Integration
* Recruiter Dashboard
* User Authentication
* Database Integration (PostgreSQL/Appwrite)
* PDF Report Generation

---

## 🎯 Use Cases

* HR Recruitment
* Talent Acquisition
* Resume Shortlisting
* Candidate Ranking
* Internship Hiring
* Technical Screening

---

## 👨‍💻 Author

**Muhammad Saad Khan**

AI Enthusiast | Machine Learning Learner | Data Analytics Enthusiast

LinkedIn: Add Your LinkedIn Profile

GitHub: Add Your GitHub Profile

---

## 📜 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for educational and commercial purposes.
