# train_model.py
import pandas as pd
import numpy as np
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# ── Download NLTK data ──
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ────────────────────────────────────────────
# 1. LOAD DATASET
# ────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('fake_job_postings.csv')

print(f"  Total rows      : {len(df)}")
print(f"  Fraudulent jobs : {df['fraudulent'].sum()}")
print(f"  Legitimate jobs : {len(df) - df['fraudulent'].sum()}")

# ────────────────────────────────────────────
# 2. COMBINE TEXT COLUMNS INTO ONE
# ────────────────────────────────────────────
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)

# ────────────────────────────────────────────
# 3. ADD INDIAN TRAFFICKING-SPECIFIC EXAMPLES
# ────────────────────────────────────────────
# These synthetic examples teach the model to catch
# trafficking-style language common in India
trafficking_phrases = [
    "free accommodation provided abroad urgent hiring",
    "urgent jobs in gulf no experience needed visa arranged",
    "work visa arranged by company travel to dubai salary 50000",
    "no documents required immediate joining overseas",
    "female candidates preferred for overseas placement agent",
    "agent will arrange everything earn lakhs per month abroad",
    "placement fee required refundable work in singapore",
    "urgent hiring from india no qualification needed dollar salary",
    "domestic worker jobs abroad free food lodging no experience",
    "earn lakhs monthly abroad no experience free accommodation",
    "contact agent for visa sponsorship work permit arranged",
    "salary in dollars marriage broker job offer abroad",
    "work in malaysia no experience needed urgent female candidates",
    "free ticket provided overseas job no documents needed",
    "gulf jobs for freshers visa and accommodation free",
]

synthetic_rows = [{'combined_text': phrase, 'fraudulent': 1}
                  for phrase in trafficking_phrases]

df_synthetic = pd.DataFrame(synthetic_rows)
df = pd.concat(
    [df[['combined_text', 'fraudulent']], df_synthetic],
    ignore_index=True
)

print(f"\nAfter adding Indian trafficking examples: {len(df)} total rows")

# ────────────────────────────────────────────
# 4. TEXT PREPROCESSING FUNCTION
# ────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(['company', 'job', 'work', 'position', 'apply', 'candidate'])

def preprocess(text):
    text = text.lower()                              # lowercase everything
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'\S+@\S+', '', text)              # remove email addresses
    text = re.sub(r'[^a-z\s]', ' ', text)            # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()         # remove extra spaces
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2       # remove stopwords + tiny words
    ]
    return ' '.join(tokens)

print("\nPreprocessing all text (this may take 1-2 minutes)...")
df['clean_text'] = df['combined_text'].apply(preprocess)
print("  Done!")

# ────────────────────────────────────────────
# 5. SPLIT INTO TRAIN AND TEST SETS
# ────────────────────────────────────────────
X = df['clean_text']
y = df['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80% train, 20% test
    random_state=42,     # makes results reproducible
    stratify=y           # keeps fraud ratio same in both splits
)

print(f"\nTraining set : {len(X_train)} samples")
print(f"Test set     : {len(X_test)} samples")

# ────────────────────────────────────────────
# 6. BUILD PIPELINE (TF-IDF + CLASSIFIER)
# ────────────────────────────────────────────
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,    # use top 10,000 words/phrases
        ngram_range=(1, 2),    # single words AND two-word phrases
        min_df=2,              # ignore terms appearing fewer than 2 times
        sublinear_tf=True,     # log-scale term frequency
    )),
    ('clf', LogisticRegression(
        class_weight='balanced',  # IMPORTANT: handles imbalanced data
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
    ))
])

# ────────────────────────────────────────────
# 7. TRAIN THE MODEL
# ────────────────────────────────────────────
print("\nTraining model...")
pipeline.fit(X_train, y_train)
print("  Training complete!")

# ────────────────────────────────────────────
# 8. EVALUATE THE MODEL
# ────────────────────────────────────────────
y_pred = pipeline.predict(X_test)

print("\n══════════════════════════════════════")
print("         MODEL EVALUATION RESULTS")
print("══════════════════════════════════════")
print(classification_report(y_test, y_pred,
      target_names=['Legitimate', 'Fraudulent']))

print("Confusion Matrix:")
print("(Rows = Actual, Columns = Predicted)")
cm = confusion_matrix(y_test, y_pred)
print(f"  True  Legit  predicted as Legit   : {cm[0][0]}")
print(f"  Legit predicted as Fraud (wrong)  : {cm[0][1]}")
print(f"  Fraud predicted as Legit (wrong)  : {cm[1][0]}")
print(f"  True  Fraud  predicted as Fraud   : {cm[1][1]}")

# ────────────────────────────────────────────
# 9. SAVE THE MODEL
# ────────────────────────────────────────────
joblib.dump(pipeline, 'model.pkl')
print("\n✅ model.pkl saved to your project folder!")
print("   You can now run predict.py")