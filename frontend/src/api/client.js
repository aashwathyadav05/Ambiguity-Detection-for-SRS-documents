/**
 * API client for the SRS Ambiguity Detector FastAPI backend
 */

const API_BASE = "http://localhost:8000";

/**
 * Health check — returns { status, model_loaded, model_path }
 */
export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

/**
 * Fetch label metadata — returns { labels: [...] }
 */
export async function fetchLabels() {
  const res = await fetch(`${API_BASE}/api/labels`);
  if (!res.ok) throw new Error(`Failed to fetch labels: ${res.status}`);
  return res.json();
}

/**
 * Analyze a document file.
 * @param {File} file — the uploaded .txt or .pdf file
 * @param {number} maxSentences — max sentences to analyze
 * @returns analysis results JSON
 */
export async function analyzeDocument(file, maxSentences = 50) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(
    `${API_BASE}/api/analyze?max_sentences=${maxSentences}`,
    { method: "POST", body: formData }
  );

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `Analysis failed: ${res.status}`);
  }

  return res.json();
}
