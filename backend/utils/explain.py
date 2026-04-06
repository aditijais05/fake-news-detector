from utils.preprocess import clean_text

SUSPICIOUS_WORDS = [
    "shocking", "breaking", "urgent", "miracle", "100%",
    "guaranteed", "secret", "exposed", "coverup", "banned",
    "censored", "they don't want", "share before deleted",
    "wake up", "deep state", "mainstream media"
]


def get_explanation(text, vectorizer=None):
    """
    Returns explainability data for a prediction.

    Two approaches:
    1. If vectorizer is passed: extract actual top TF-IDF features
       (the words that had the most weight in the model's decision)
    2. Fallback: keyword matching against known suspicious phrases
    """

    # ---- Approach 1: Real TF-IDF Feature Importance ----
    if vectorizer is not None:
        try:
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])
            feature_names = vectorizer.get_feature_names_out()

            # Get (word, tfidf_score) pairs for this specific input
            scores = zip(feature_names, vec.toarray()[0])

            # Keep only words that actually appeared (score > 0)
            # Sort by score descending to get most important words first
            top_features = sorted(
                [(word, round(float(score), 4)) for word, score in scores if score > 0],
                key=lambda x: x[1],
                reverse=True
            )[:10]  # top 10 influential words

            top_words = [w for w, s in top_features]

        except Exception:
            top_words = []
            top_features = []
    else:
        top_words = []
        top_features = []

    # ---- Approach 2: Suspicious keyword matching ----
    text_lower = text.lower()
    found_suspicious = [
        word for word in SUSPICIOUS_WORDS
        if word in text_lower
    ]

    # ---- Risk level based on suspicious word count ----
    if len(found_suspicious) >= 3:
        risk = "High"
        reason = f"Contains {len(found_suspicious)} known misinformation phrases"
    elif len(found_suspicious) >= 1:
        risk = "Medium"
        reason = f"Contains {len(found_suspicious)} suspicious phrase(s)"
    else:
        risk = "Low"
        reason = "No known suspicious phrases detected"

    return {
        "top_tfidf_features": top_words,          # actual ML feature importance
        "suspicious_phrases": found_suspicious,    # rule-based keyword matches
        "risk_level": risk,
        "reason": reason,
        "word_count": len(text.split())
    }