import React, { useState } from "react";

function InputBox({ onAnalyze }) {
  const [text, setText] = useState("");

  return (
    <div>
      <textarea
        rows="6"
        placeholder="Paste news here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <br />
      <button onClick={() => onAnalyze(text)}>Analyze</button>
    </div>
  );
}

export default InputBox;