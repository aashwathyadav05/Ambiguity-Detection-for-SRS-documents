export default function Hero() {
  return (
    <div className="hero">
      <h1>SRS Ambiguity Detector</h1>
      <p>
        Automatically classify linguistic ambiguities in Software Requirements
        Specification documents using a fine-tuned RoBERTa model combined with
        rule-based heuristics.
      </p>
      <div className="hero-badges">
        <span className="badge">RoBERTa</span>
        <span className="badge">NLP</span>
        <span className="badge">6-Class Classification</span>
        <span className="badge">Rule-Based Engine</span>
      </div>
    </div>
  );
}
