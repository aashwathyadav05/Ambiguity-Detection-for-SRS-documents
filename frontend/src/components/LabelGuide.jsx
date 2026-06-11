const LABELS = [
  {
    name: "Lexical",
    color: "#eab308",
    desc: "A word has multiple meanings (e.g., 'process', 'handle', 'light').",
  },
  {
    name: "Syntactic",
    color: "#f97316",
    desc: "Sentence structure allows multiple parse trees.",
  },
  {
    name: "Semantic",
    color: "#a855f7",
    desc: "Meaning is unclear even with a fixed parse tree.",
  },
  {
    name: "Syntax",
    color: "#fb923c",
    desc: "Structural/grammatical issue causing misinterpretation.",
  },
  {
    name: "Pragmatic",
    color: "#f43f5e",
    desc: "Context-dependent meaning; intent unclear without extra knowledge.",
  },
  {
    name: "Clean",
    color: "#22c55e",
    desc: "Requirement is clear and unambiguous.",
  },
];

export default function LabelGuide() {
  return (
    <section className="label-guide" id="label-guide">
      <h2 className="section-header">
        <span className="icon">📋</span>
        Label Guide
      </h2>
      <div className="label-guide-grid">
        {LABELS.map((label) => (
          <div key={label.name} className="label-card">
            <div
              className="label-dot"
              style={{ backgroundColor: label.color }}
            />
            <div className="label-card-content">
              <div
                className="label-card-name"
                style={{ color: label.color }}
              >
                {label.name}
              </div>
              <div className="label-card-desc">{label.desc}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
