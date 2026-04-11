from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
from utils.preprocess import clean_text
from utils.explain import get_explanation
from transformers import pipeline
import time
import os

# Lazy load - model loads only on first request
bert_model = None

def get_bert_model():
    global bert_model
    if bert_model is None:
        bert_model = pipeline(
            "text-classification",
            model="hamzab/roberta-fake-news-classification",
            truncation=True,
            max_length=512
        )
    return bert_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "model", "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "model", "vectorizer.pkl"), "rb"))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "models": ["Logistic Regression", "RoBERTa"]}), 200

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Missing required field: 'text'"}), 400
    data = request.json['text'].strip()
    if len(data) < 10:
        return jsonify({"error": "Text too short."}), 400
    if len(data) > 10000:
        data = data[:10000]

    cleaned = clean_text(data)
    vector = vectorizer.transform([cleaned])
    lr_pred = model.predict(vector)[0]
    lr_proba = model.predict_proba(vector)[0]
    lr_confidence = lr_proba.max()
    lr_label = "Real" if lr_pred == 1 else "Fake"

    bert_result = get_bert_model()(data[:512])[0]
    bert_raw_label = bert_result['label']
    bert_confidence = round(bert_result['score'] * 100, 2)
    bert_label = "Real" if bert_raw_label == "LABEL_1" else "Fake"

    lr_fake_prob = lr_proba[0] if lr_pred == 0 else (1 - lr_confidence)
    bert_fake_prob = bert_result['score'] if bert_raw_label == "LABEL_0" else (1 - bert_result['score'])
    ensemble_fake_prob = (lr_fake_prob * 0.35) + (bert_fake_prob * 0.65)
    ensemble_confidence = round(max(ensemble_fake_prob, 1 - ensemble_fake_prob) * 100, 2)

    if ensemble_fake_prob > 0.65:
        final_verdict = "Fake"
    elif ensemble_fake_prob > 0.40:
        final_verdict = "Misleading"
    else:
        final_verdict = "Real"

    explanation = get_explanation(data, vectorizer)
    elapsed_ms = round((time.time() - start_time) * 1000)

    return jsonify({
        "verdict": final_verdict,
        "confidence": ensemble_confidence,
        "models_disagree": lr_label != bert_label,
        "lr": {"result": lr_label, "confidence": round(lr_confidence * 100, 2)},
        "bert": {"result": bert_label, "confidence": bert_confidence},
        "explanation": explanation,
        "meta": {"processing_time_ms": elapsed_ms, "text_length": len(data)}
    }), 200

@app.route('/feedback', methods=['POST'])
def feedback():
    if not request.json:
        return jsonify({"error": "Missing JSON body"}), 400
    required = ['text', 'predicted', 'correct']
    for field in required:
        if field not in request.json:
            return jsonify({"error": f"Missing field: {field}"}), 400
    import json
    entry = {
        "text": request.json['text'][:500],
        "predicted": request.json['predicted'],
        "correct": request.json['correct'],
        "timestamp": time.time()
    }
    with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return jsonify({"status": "saved"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)