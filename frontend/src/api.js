const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

export async function analyzeNews(text) {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.error || "API Error");
  }

  return data;
}