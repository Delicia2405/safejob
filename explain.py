import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer
from predict import predict

# Load model
model = joblib.load("model.pkl")

# Class labels
class_names = ["Legitimate", "Fraudulent"]

# Create explainer
explainer = LimeTextExplainer(class_names=class_names)


def explain(job_description: str):
    """
    Returns important words with scores + reason
    """

    if not job_description or not job_description.strip():
        return []

    # LIME needs probability function
    def predict_proba(texts):
        probs = []
        for text in texts:
            fraud_prob = predict(text)
            probs.append([1 - fraud_prob, fraud_prob])
        return np.array(probs)

    exp = explainer.explain_instance(
        job_description,
        predict_proba,
        num_features=10
    )

    results = []

    for word, score in exp.as_list():
        word = str(word)

        # ❌ Remove useless words
        if word.lower() in ["in", "for", "the", "and", "job", "no"]:
            continue

        # ❌ Remove weak signals
        if abs(score) < 0.01:
            continue

        label = "risky" if score > 0 else "safe"

        # ✅ Add reason (VERY IMPORTANT FOR DEMO)
        if label == "risky":
            if "free" in word.lower():
                reason = "lure tactic"
            elif "visa" in word.lower():
                reason = "immigration trap"
            elif "urgent" in word.lower():
                reason = "pressure tactic"
            elif "girls" in word.lower():
                reason = "targeted recruitment"
            elif "dubai" in word.lower():
                reason = "high-risk destination"
            else:
                reason = "suspicious wording"
        else:
            reason = "normal job term"

        results.append({
            "word": word,
            "score": round(score, 3),
            "type": label,
            "reason": reason
        })

    return results


# ✅ TEST BLOCK
if __name__ == "__main__":
    text = "Urgent job in Dubai free visa no experience for girls"
    
    print("\n🔍 Explanation:\n")
    explanation = explain(text)

    for item in explanation:
        print(item)