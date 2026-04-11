from utils.preprocess import clean_text

SUSPICIOUS_WORDS = [
    "shocking", "breaking", "urgent", "miracle", "100%",
    "guaranteed", "secret", "exposed", "coverup", "banned",
    "censored", "they don't want", "share before deleted",
    "wake up", "deep state", "mainstream media"
]

def get_explanation(text, vectorizer=None):
    if vectorizer is not None:
        try:
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])
            feature_names = vectorizer.get_feature_names_out()
            scores = zip(feature_names, vec.toarray()[0])
            top_features = sorted(
                [(word, round(float(score), 4)) for word, score in scores if score > 0],
                key=lambda x: x[1], reverse=True
            )[:10]
            top_words = [w for w, s in top_features]
        except Exception:
            top_words = []
    else:
        top_words = []

    text_lower = text.lower()
    found_suspicious = [w for w in SUSPICIOUS_WORDS if w in text_lower]

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
        "top_tfidf_features": top_words,
        "suspicious_phrases": found_suspicious,
        "risk_level": risk,
        "reason": reason,
        "word_count": len(text.split())
    }