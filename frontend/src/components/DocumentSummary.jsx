const LABEL_ORDER = ["Lexical", "Syntactic", "Semantic", "Syntax", "Pragmatic", "Clean"];

const COLORS = {
  Lexical:   "#eab308",
  Syntactic: "#f97316",
  Semantic:  "#a855f7",
  Syntax:    "#fb923c",
  Pragmatic: "#f43f5e",
  Clean:     "#22c55e",
};

export default function DocumentSummary({ summary, totalSentences, analyzedSentences, filename }) {
  if (!summary) return null;

  return (
    <section className="summary-section" id="document-summary">
      <h2 className="section-header">
        <span className="icon">📊</span>
        Document Summary
      </h2>
      <p className="results-info">
        Analyzed <strong>{analyzedSentences}</strong> of{" "}
        <strong>{totalSentences}</strong> sentences from{" "}
        <strong>{filename}</strong>
      </p>
      <div className="summary-grid">
        {LABEL_ORDER.map((label) => (
          <div key={label} className="metric-box">
            <div
              className="metric-value"
              style={{ color: COLORS[label] }}
            >
              {summary[label] ?? 0}
            </div>
            <div className="metric-label">{label}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
