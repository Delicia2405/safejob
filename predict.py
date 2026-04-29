# predict.py
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data silently if not already downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ── Load model ONCE when file is imported ──
# This means Person 4's web app only loads it once, not on every request
_model = joblib.load('model.pkl')
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))
_stop_words.update(['company', 'job', 'work', 'position', 'apply', 'candidate'])


def _preprocess(text: str) -> str:
    """
    Internal function — cleans text the same way train_model.py does.
    Must stay identical to the preprocess() in train_model.py.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words and len(t) > 2
    ]
    return ' '.join(tokens)


def predict(job_description: str) -> float:
    """
    Predict how likely a job posting is fraudulent.

    Args:
        job_description (str): Any job posting text — title, description,
                                requirements, etc. Can be raw/unclean.

    Returns:
        float: A score from 0.0 to 1.0
               0.0 = almost certainly legitimate
               1.0 = almost certainly fraudulent
               > 0.5 = treat as suspicious / flag for review

    Example:
        from predict import predict

        score = predict("Urgent jobs in Dubai no experience free visa arranged")
        print(score)   # e.g. 0.91

        score = predict("Software Engineer at Infosys, 3 years Python required")
        print(score)   # e.g. 0.04
    """
    # Handle empty input
    if not job_description or not job_description.strip():
        return 0.0

    # Preprocess exactly like training data
    clean = _preprocess(job_description)

    # Handle edge case where cleaning removes all words
    if not clean:
        return 0.0

    # Get probability of class 1 (fraudulent)
    proba = _model.predict_proba([clean])[0][1]
    return round(float(proba), 4)


# ── Self-test: runs only when you do  python predict.py  directly ──
if __name__ == '__main__':
    print("═══════════════════════════════════════════════")
    print("         SafeJob — predict.py self-test")
    print("═══════════════════════════════════════════════\n")

    test_cases = [
        ("SHOULD BE HIGH",  "Urgent jobs in Dubai no experience needed free accommodation visa arranged agent"),
        ("SHOULD BE HIGH",  "Female candidates wanted overseas placement earn lakhs no documents needed free ticket"),
        ("SHOULD BE HIGH",  "Gulf jobs freshers salary dollars visa sponsorship agent arrange everything"),
        ("SHOULD BE LOW",   "Software Engineer Infosys Bangalore 3 years Python experience required"),
        ("SHOULD BE LOW",   "Data Analyst role must have SQL Excel skills apply resume company portal"),
        ("SHOULD BE LOW",   "Marketing Manager 5 years experience FMCG sector Mumbai competitive package"),
    ]

    for expectation, text in test_cases:
        score = predict(text)
        if score > 0.5:
            flag = "🚨 FLAGGED AS FRAUD"
        else:
            flag = "✅ LOOKS LEGITIMATE"

        print(f"Expectation : {expectation}")
        print(f"Result      : {flag}  (score: {score})")
        print(f"Text        : {text[:75]}...")
        print()