import React from "react";

function highlightText(text, phrases) {
  let result = text;

  phrases.forEach((word) => {
    const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`(${escaped})`, "gi");
    result = result.replace(
      regex,
      `<span class="highlight">$1</span>`
    );
  });

  return result;
}

function ResultCard({ result }) {
  const { verdict, confidence, bert, lr, explanation } = result;

  const color =
    verdict === "Fake"
      ? "red"
      : verdict === "Real"
      ? "green"
      : "orange";

  return (
    <div className="result-card" style={{ borderColor: color }}>
      
      {/* MAIN VERDICT */}
      <h1 style={{ color }}>{verdict.toUpperCase()} 🚨</h1>
      <p>Confidence: {confidence}%</p>

      <div className="progress">
        <div
          className="progress-bar"
          style={{ width: `${confidence}%`, background: color }}
        />
      </div>

      <p><b>Risk Level:</b> {explanation.risk_level}</p>

      {/* MODEL OUTPUT */}
      <div>
        <p>🤖 BERT: {bert.result} ({bert.confidence}%)</p>
        <p>📊 LR: {lr.result} ({lr.confidence}%)</p>
      </div>

      {/* WARNING */}
      {result.models_disagree && (
        <p className="warn-text">⚠ Models disagree</p>
      )}

      {/* SUSPICIOUS PHRASES */}
      <div>
        <b>⚠ Suspicious Phrases:</b>
        <ul>
          {explanation.suspicious_phrases?.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      </div>

      {/* HIGHLIGHTED TEXT */}
      <div
        dangerouslySetInnerHTML={{
          __html: highlightText(
            result.original_text,
            explanation.suspicious_phrases || []
          ),
        }}
      />
    </div>
  );
}

export default ResultCard;