import React, { useState } from "react";
import { analyzeNews } from "./api";
import ResultCard from "./components/ResultCard";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError("");

    try {
      const data = await analyzeNews(text);
      setResult(data);
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <h1>📰 Fake News Detector</h1>

      <textarea
        rows="5"
        placeholder="Paste news text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <p></p>

      <button onClick={handleAnalyze}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && <ResultCard result={result} />}
    </div>
  );
}

export default App;