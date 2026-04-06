from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
from utils.preprocess import clean_text
from utils.explain import get_explanation
from transformers import pipeline
import time

# ----------------------------------------------------------------
# Load Models at startup — NOT inside the route function.
# Loading inside the route would reload the model on every request
# which takes 10+ seconds each time.
# ----------------------------------------------------------------

# RoBERTa fine-tuned specifically on fake news
# LABEL_0 = Fake, LABEL_1 = Real (confirmed from model card)
bert_model = pipeline(
    "text-classification",
    model="hamzab/roberta-fake-news-classification",
    truncation=True,      # auto-truncate inputs longer than 512 tokens
    max_length=512
)

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


# ----------------------------------------------------------------
# Health Check — always add this to every API you build.
# Load balancers and monitoring tools ping this to check if
# the server is alive without running expensive ML inference.
# ----------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "models": ["Logistic Regression", "RoBERTa (hamzab/roberta-fake-news-classification)"]
    }), 200


# ----------------------------------------------------------------
# Main Prediction Endpoint
# ----------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    data_json = request.get_json(force=True, silent=True)

    if not data_json or 'text' not in data_json:
        return jsonify({"error": "Missing required field: 'text'"}), 400

    data = data_json['text'].strip()

    if len(data) < 10:
        return jsonify({"error": "Text too short"}), 400

    truncated = False
    if len(data) > 10000:
        data = data[:10000]
        truncated = True

    # -------- Logistic Regression --------
    cleaned = clean_text(data)
    vector = vectorizer.transform([cleaned])

    lr_pred = model.predict(vector)[0]
    lr_proba = model.predict_proba(vector)[0]

    lr_classes = list(model.classes_)
    fake_index = lr_classes.index(0)

    lr_fake_prob = lr_proba[fake_index]
    lr_confidence = round(max(lr_proba) * 100, 2)
    lr_label = "Fake" if lr_fake_prob >= 0.5 else "Real"

    # -------- BERT --------
    bert_out = bert_model(data[:512])[0]

    if bert_out['label'] == "LABEL_0":
        bert_fake_prob = bert_out['score']
        bert_label = "Fake"
    else:
        bert_fake_prob = 1 - bert_out['score']
        bert_label = "Real"

    bert_confidence = round(max(bert_out['score'], 1 - bert_out['score']) * 100, 2)

    # -------- Ensemble --------
    ensemble_fake_prob = (0.65 * bert_fake_prob) + (0.35 * lr_fake_prob)

    if ensemble_fake_prob >= 0.7:
        final_verdict = "Fake"
    elif ensemble_fake_prob <= 0.3:
        final_verdict = "Real"
    else:
        final_verdict = "Misleading"

    models_disagree = lr_label != bert_label

    ensemble_confidence = round(
        max(ensemble_fake_prob, 1 - ensemble_fake_prob) * 100, 2
    )

    # boost confidence if both agree strongly
    if not models_disagree and bert_confidence > 90 and lr_confidence > 80:
        ensemble_confidence = min(100, ensemble_confidence + 5)

    # -------- Explanation --------
    explanation = get_explanation(data, vectorizer)

    if ensemble_fake_prob > 0.8:
        explanation["risk_level"] = "High"
    elif ensemble_fake_prob > 0.5:
        explanation["risk_level"] = "Medium"
    else:
        explanation["risk_level"] = "Low"

    elapsed_ms = round((time.time() - start_time) * 1000)

    return jsonify({
        "verdict": final_verdict,
        "confidence": ensemble_confidence,
        "models_disagree": models_disagree,

        "original_text": data,

        "lr": {
            "result": lr_label,
            "confidence": lr_confidence
        },
        "bert": {
            "result": bert_label,
            "confidence": bert_confidence
        },

        "explanation": explanation,

        "meta": {
            "processing_time_ms": elapsed_ms,
            "text_length": len(data),
            "truncated": truncated
        }
    }), 200

# ----------------------------------------------------------------
# Feedback Endpoint
# Lets users tell you when the model was wrong.
# This data can be used to retrain and improve the model later.
# Having this shows you understand the full ML lifecycle.
# ----------------------------------------------------------------
@app.route('/feedback', methods=['POST'])
def feedback():
    if not request.json:
        return jsonify({"error": "Missing JSON body"}), 400

    required = ['text', 'predicted', 'correct']
    for field in required:
        if field not in request.json:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Append to a local JSONL file (one JSON object per line)
    # In production you'd write to a database instead
    import json
    entry = {
        "text": request.json['text'][:500],   # store first 500 chars only
        "predicted": request.json['predicted'],
        "correct": request.json['correct'],
        "timestamp": time.time()
    }

    with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return jsonify({"status": "saved", "message": "Thank you for the feedback!"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)