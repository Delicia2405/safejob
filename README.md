# 🛡️ SafeJob — AI-Powered Fake Job Detection for Migrant Workers

SafeJob uses artificial intelligence to detect fake job offers targeting vulnerable migrant workers in India — before it's too late.

## 🎯 What It Does
Migrant workers paste a suspicious job offer into SafeJob. Within seconds, our AI analyzes it across 4 specialized modules and returns a risk score from **LOW** to **CRITICAL**.

## ⚡ 4 AI Modules

| Module | What It Does |
| :--- | :--- |
| **🤖 NLP Classifier** | Trained on 17,880 real/fake job postings using TF-IDF + Logistic Regression. |
| **🔍 XAI Word Highlights** | LIME (Explainable AI) shows exactly which words/phrases triggered the alert. |
| **💰 Salary Anomaly Engine** | Z-score analysis compares offered salary vs. Indian market averages. |
| **📍 Location Risk** | NCRB crime data scores trafficking risk by city and state. |

---

## ⚖️ Combining Formula
To ensure accuracy, the final verdict is calculated using a weighted average:

$$Final Score = (0.40 \times NLP) + (0.20 \times Salary) + (0.40 \times Location)$$

| Score | Verdict |
| :--- | :--- |
| **≥ 0.75** | 🔴 **CRITICAL** |
| **≥ 0.55** | 🟠 **HIGH** |
| **≥ 0.35** | 🟡 **MEDIUM** |
| **< 0.35** | 🟢 **LOW** |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Git

### Installation
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/yourusername/safejob.git](https://github.com/yourusername/safejob.git)
   cd safejob
Set up Virtual Environment:

Bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate
Install Dependencies:

Bash
pip install flask scikit-learn nltk lime numpy pandas folium
Download NLTK Data:

Bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
Run the App:

Bash
python app.py
Open http://127.0.0.1:5000 in your browser.

🗂️ Project Structure
Plaintext
safejob/
├── app.py              # Flask web application & routing
├── predict.py          # NLP model prediction logic
├── explain.py          # LIME explainability implementation
├── salary_engine.py    # Salary anomaly detection
├── heatmap.py          # Location risk scoring
├── train_model.py      # Model retraining script
├── model.pkl           # Trained ML model
├── vectorizer.pkl      # TF-IDF vectorizer
├── reports.db          # SQLite database for user-reported scams
└── templates/          # HTML Frontend
    ├── index.html      # Landing page
    ├── check.html      # Job input form
    ├── result_score.html # Step 1: Overall Risk
    └── ...             # Module-specific results
🧪 Test Cases
✅ Safe Job: Title: Software Engineer | Salary: ₹45,000 | City: Pune → LOW/MEDIUM

🚨 Risky Job: Title: Overseas Helper | Salary: ₹1,50,000 | City: Delhi → HIGH/CRITICAL

🚨 Emergency Helpline
National Anti-Trafficking Helpline: 1800-419-8588
Free · Confidential · Available 24/7

⚠️ Disclaimer
SafeJob is a PBL academic project. It is not a substitute for professional legal advice. Always verify job offers independently.

👥 Team
Built with ❤️ by a team of 4 as a final year project to combat human trafficking traps disguised as employment.
