import { useState } from "react";

const COLORS = {
  Lexical:   "#eab308",
  Syntactic: "#f97316",
  Semantic:  "#a855f7",
  Syntax:    "#fb923c",
  Pragmatic: "#f43f5e",
  Clean:     "#22c55e",
};

const BG_COLORS = {
  Lexical:   "#fefce8",
  Syntactic: "#fff7ed",
  Semantic:  "#fdf4ff",
  Syntax:    "#fff7ed",
  Pragmatic: "#fff1f2",
  Clean:     "#f0fdf4",
};

const TEXT_COLORS = {
  Lexical:   "#713f12",
  Syntactic: "#7c2d12",
  Semantic:  "#581c87",
  Syntax:    "#7c2d12",
  Pragmatic: "#881337",
  Clean:     "#14532d",
};

const LABEL_ORDER = ["Lexical", "Syntactic", "Semantic", "Syntax", "Pragmatic", "Clean"];

export default function SentenceResult({ result, index }) {
  const {
    sentence,
    model_label,
    model_confidence,
    model_probabilities,
    rule_label,
    rule_flags,
  } = result;

  const isClean = model_label === "Clean" && rule_label === "Clean";
  const [expanded, setExpanded] = useState(!isClean);

  const color = COLORS[model_label] || COLORS.Clean;
  const bgColor = BG_COLORS[model_label] || BG_COLORS.Clean;
  const textColor = TEXT_COLORS[model_label] || TEXT_COLORS.Clean;

  const ruleColor = COLORS[rule_label] || COLORS.Clean;
  const ruleBg = BG_COLORS[rule_label] || BG_COLORS.Clean;
  const ruleTextColor = TEXT_COLORS[rule_label] || TEXT_COLORS.Clean;

  const preview =
    sentence.length > 90 ? sentence.slice(0, 90) + "…" : sentence;

  // Sort probabilities descending
  const sortedProbs = LABEL_ORDER
    .map((label) => ({
      label,
      prob: model_probabilities?.[label] ?? 0,
    }))
    .sort((a, b) => b.prob - a.prob);

  return (
    <div
      className="result-card"
      style={{ animationDelay: `${index * 30}ms` }}
    >
      {/* Header (always visible) */}
      <div
        className="result-card-header"
        onClick={() => setExpanded((prev) => !prev)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) =>
          (e.key === "Enter" || e.key === " ") &&
          setExpanded((prev) => !prev)
        }
        aria-expanded={expanded}
        id={`result-header-${index}`}
      >
        <div
          className="result-card-indicator"
          style={{ backgroundColor: color }}
        />
        <div className="result-card-labels">
          <span
            className="result-pill"
            style={{
              backgroundColor: bgColor,
              color: textColor,
            }}
          >
            Model: {model_label}
          </span>
          <span
            className="result-pill"
            style={{
              backgroundColor: ruleBg,
              color: ruleTextColor,
            }}
          >
            Rule: {rule_label}
          </span>
        </div>
        <span className="result-sentence-preview">{preview}</span>
        <span className={`result-chevron ${expanded ? "open" : ""}`}>
          ▼
        </span>
      </div>

      {/* Expandable body */}
      <div className={`result-card-body ${expanded ? "expanded" : ""}`}>
        {/* Full sentence */}
        <div
          className="result-sentence-full"
          style={{
            backgroundColor: bgColor,
            borderColor: color,
            color: textColor,
          }}
        >
          {sentence}
        </div>

        {/* Confidence bar */}
        {model_label !== "N/A" && (
          <div className="confidence-section">
            <div className="confidence-label">
              Model Confidence
            </div>
            <div className="confidence-bar-bg">
              <div
                className="confidence-bar-fill"
                style={{
                  width: `${(model_confidence * 100).toFixed(1)}%`,
                  backgroundColor: color,
                }}
              />
            </div>
            <div className="confidence-value">
              {(model_confidence * 100).toFixed(1)}%
            </div>
          </div>
        )}

        {/* Two-column detail */}
        <div className="result-detail-grid">
          {/* Probability distribution */}
          <div>
            <div className="detail-column-title">
              Model Probability Distribution
            </div>
            {sortedProbs.map(({ label, prob }) => (
              <div key={label} className="prob-item">
                <span className="prob-label">{label}</span>
                <div className="prob-bar-bg">
                  <div
                    className="prob-bar-fill"
                    style={{
                      width: `${(prob * 100).toFixed(1)}%`,
                      backgroundColor: COLORS[label],
                    }}
                  />
                </div>
                <span className="prob-value">
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>

          {/* Rule-based flags */}
          <div>
            <div className="detail-column-title">
              Rule-Based Engine (Class: {rule_label})
            </div>
            {rule_flags && rule_flags.length > 0 ? (
              <ul className="rule-flags-list">
                {rule_flags.map((flag, i) => (
                  <li key={i} className="rule-flag-item">
                    {flag}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="no-flags">No heuristic flags detected</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
