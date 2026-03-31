"""
Ambiguity Detection for SRS Documents
Streamlit Demo App — powered by fine-tuned RoBERTa
"""

import streamlit as st
import torch
import re
from pathlib import Path
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pdfplumber

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SRS Ambiguity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f172a 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #1e40af33;
}
.hero h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
}
.hero .badge {
    display: inline-block;
    background: #1e40af44;
    border: 1px solid #3b82f6;
    color: #93c5fd;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 999px;
    margin-right: 6px;
    margin-top: 10px;
}

/* Result card */
.result-card {
    padding: 1.2rem 1.4rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    border-left: 5px solid;
    font-size: 0.92rem;
}
.result-clean     { background:#f0fdf4; border-color:#22c55e; color:#14532d; }
.result-lexical   { background:#fefce8; border-color:#eab308; color:#713f12; }
.result-syntactic { background:#fff7ed; border-color:#f97316; color:#7c2d12; }
.result-semantic  { background:#fdf4ff; border-color:#a855f7; color:#581c87; }
.result-pragmatic { background:#fff1f2; border-color:#f43f5e; color:#881337; }
.result-syntax    { background:#fff7ed; border-color:#fb923c; color:#7c2d12; }

.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.result-text {
    font-size: 0.88rem;
    opacity: 0.85;
}
.confidence-bar-bg {
    height: 6px;
    border-radius: 3px;
    background: rgba(0,0,0,0.08);
    margin-top: 8px;
}
.confidence-bar-fill {
    height: 6px;
    border-radius: 3px;
}

/* Sentence pill */
.pill {
    display: inline-block;
    background: #1e293b;
    color: #e2e8f0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 999px;
    margin-bottom: 8px;
}

/* Metrics row */
.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-box .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #0f172a;
}
.metric-box .label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 2px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
LABEL_NAMES = {
    0: "Lexical",
    1: "Syntactic",
    2: "Semantic",
    3: "Syntax",
    4: "Pragmatic",
    5: "Clean",
}

LABEL_CSS = {
    "Lexical":    "lexical",
    "Syntactic":  "syntactic",
    "Semantic":   "semantic",
    "Syntax":     "syntax",
    "Pragmatic":  "pragmatic",
    "Clean":      "clean",
}

LABEL_EMOJI = {
    "Lexical":   "🔤",
    "Syntactic": "🧩",
    "Semantic":  "💬",
    "Syntax":    "📐",
    "Pragmatic": "🗣️",
    "Clean":     "✅",
}

LABEL_DESC = {
    "Lexical":   "A word has multiple meanings (e.g., 'process', 'handle', 'light').",
    "Syntactic": "Sentence structure allows multiple parse trees.",
    "Semantic":  "Meaning is unclear even with a fixed parse tree.",
    "Syntax":    "Structural/grammatical issue causing misinterpretation.",
    "Pragmatic": "Context-dependent meaning; intent unclear without extra knowledge.",
    "Clean":     "Requirement is clear and unambiguous.",
}

BAR_COLORS = {
    "Lexical":   "#eab308",
    "Syntactic": "#f97316",
    "Semantic":  "#a855f7",
    "Syntax":    "#fb923c",
    "Pragmatic": "#f43f5e",
    "Clean":     "#22c55e",
}

# ─────────────────────────────────────────────
#  Model path (hardcoded)
# ─────────────────────────────────────────────
MODEL_PATH = "models/roberta-ambiguity-final"


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
VAGUE_WORDS     = ["adequate", "appropriate", "fast", "flexible", "high", "large",
                   "maximize", "minimize", "optimal", "quick", "recent", "robust",
                   "simple", "small", "sufficient", "user-friendly", "various",
                   "several", "many", "easy", "efficient", "modern", "normal"]
MODAL_VERBS     = ["may", "might", "could", "should", "shall", "must", "will",
                   "would", "can"]
PRONOUNS        = ["it", "they", "them", "this", "that", "these", "those",
                   "its", "their"]
PASSIVE_RE      = re.compile(
    r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', re.I
)
ATTACHMENT_RE   = re.compile(
    r'\b(with|that|which|who|where)\b.*\b(and|or)\b', re.I
)

def rule_based_flags(sentence: str) -> list[str]:
    s = sentence.lower()
    flags = []
    found_vague = [w for w in VAGUE_WORDS if re.search(rf'\b{w}\b', s)]
    if found_vague:
        flags.append(f"Vague quantifier(s): *{', '.join(found_vague)}*")
    found_modals = [w for w in MODAL_VERBS if re.search(rf'\b{w}\b', s)]
    if found_modals:
        flags.append(f"Ambiguous modal verb(s): *{', '.join(found_modals)}*")
    found_pro = [w for w in PRONOUNS if re.search(rf'\b{w}\b', s)]
    if found_pro:
        flags.append(f"Unclear pronoun reference(s): *{', '.join(found_pro)}*")
    if PASSIVE_RE.search(sentence):
        flags.append("Passive voice detected — actor unspecified.")
    if ATTACHMENT_RE.search(sentence):
        flags.append("Possible modifier attachment ambiguity.")
    return flags


# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model     = RobertaForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict_sentence(sentence: str, tokenizer, model, threshold: float = 0.0):
    inputs  = tokenizer(sentence, return_tensors="pt",
                        truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id = int(torch.argmax(logits))
    label   = LABEL_NAMES[pred_id]
    conf    = probs[pred_id]
    return label, conf, {LABEL_NAMES[i]: p for i, p in enumerate(probs)}


# ─────────────────────────────────────────────
#  Text → sentences
# ─────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    # Simple sentence splitter on ., !, ? or newlines
    raw = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 8]


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.05,
        help="Predictions below this confidence will be flagged with a warning."
    )

    show_all_probs = st.checkbox("Show full probability distribution", value=False)
    show_rules     = st.checkbox("Show rule-based heuristic hints", value=True)

    st.markdown("---")
    st.markdown("### 📋 Label Guide")
    for lbl, desc in LABEL_DESC.items():
        st.markdown(f"**{LABEL_EMOJI[lbl]} {lbl}**  \n{desc}")

    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.75rem;color:#475569;'>Fine-tuned RoBERTa · "
        "Fault-prone SRS Dataset · MIT License</span>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 SRS Ambiguity Detector</h1>
  <p>Automatically classify linguistic ambiguities in Software Requirements Specification documents.</p>
  <span class="badge">RoBERTa</span>
  <span class="badge">NLP4RE</span>
  <span class="badge">6-class</span>
  <span class="badge">Fault-prone SRS Dataset</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
model_loaded = False
tokenizer, model = None, None

if Path(MODEL_PATH).exists():
    with st.spinner("Loading model…"):
        try:
            tokenizer, model = load_model(MODEL_PATH)
            model_loaded = True
            st.success(f"✅ Model loaded from `{MODEL_PATH}`", icon="🤖")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
else:
    st.warning(
        f"⚠️ Model not found at `{MODEL_PATH}`. "
        "Rule-based heuristics are still available below.",
        icon="📂",
    )


# ─────────────────────────────────────────────
#  Input tabs
# ─────────────────────────────────────────────
tab_single, tab_bulk, tab_file = st.tabs(
    ["✏️ Single Requirement", "📋 Bulk Requirements", "📄 Upload SRS File"]
)


# ── Tab 1: Single sentence ───────────────────
with tab_single:
    st.markdown("#### Enter a requirement sentence")
    example_opts = [
        "The system should respond quickly to user requests.",
        "The software shall process all transactions reliably.",
        "The module must handle it efficiently when the server fails.",
        "The application shall support various input formats.",
        "The login page must display an error message if authentication fails.",
    ]
    use_example = st.selectbox("Or pick an example:", ["— type your own —"] + example_opts)
    default_val = "" if use_example == "— type your own —" else use_example
    sentence    = st.text_area("Requirement text", value=default_val, height=100,
                               placeholder="The system shall respond to user input within 2 seconds…")

    if st.button("Analyze", type="primary", key="btn_single"):
        if not sentence.strip():
            st.warning("Please enter a requirement sentence.")
        else:
            flags = rule_based_flags(sentence) if show_rules else []

            if model_loaded:
                label, conf, all_probs = predict_sentence(sentence, tokenizer, model)
                css_cls = LABEL_CSS[label]
                bar_clr = BAR_COLORS[label]
                low_conf = conf < confidence_threshold

                st.markdown(f"""
                <div class="result-card result-{css_cls}">
                  <div class="result-label">{LABEL_EMOJI[label]} {label} Ambiguity
                    {'&nbsp;&nbsp;<span style="font-size:0.75rem;opacity:0.7">⚠️ low confidence</span>' if low_conf else ''}
                  </div>
                  <div class="result-text">{LABEL_DESC[label]}</div>
                  <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{conf*100:.1f}%;background:{bar_clr};"></div>
                  </div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;margin-top:4px;opacity:0.7;">
                    Confidence: {conf*100:.1f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if show_all_probs:
                    st.markdown("**Probability distribution**")
                    sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
                    for lbl, p in sorted_probs:
                        st.progress(p, text=f"{LABEL_EMOJI[lbl]} {lbl}: {p*100:.1f}%")

            else:
                st.info("Model not loaded — showing rule-based analysis only.", icon="ℹ️")

            if show_rules:
                if flags:
                    st.markdown("**🔎 Heuristic hints:**")
                    for f in flags:
                        st.markdown(f"- {f}")
                else:
                    st.markdown("_No heuristic flags triggered._")


# ── Tab 2: Bulk ──────────────────────────────
with tab_bulk:
    st.markdown("#### Paste multiple requirements (one per line)")
    bulk_text = st.text_area(
        "Requirements",
        height=200,
        placeholder="The system shall…\nThe module must…\nThe application should…",
    )

    if st.button("Analyze All", type="primary", key="btn_bulk"):
        lines = [l.strip() for l in bulk_text.splitlines() if len(l.strip()) > 8]
        if not lines:
            st.warning("Please enter at least one requirement.")
        else:
            results = []
            prog = st.progress(0, text="Analyzing…")
            for i, line in enumerate(lines):
                if model_loaded:
                    label, conf, _ = predict_sentence(line, tokenizer, model)
                else:
                    label, conf = "N/A (no model)", 0.0
                flags = rule_based_flags(line) if show_rules else []
                results.append((line, label, conf, flags))
                prog.progress((i + 1) / len(lines), text=f"Analyzing {i+1}/{len(lines)}…")
            prog.empty()

            # Summary metrics
            if model_loaded:
                from collections import Counter
                counts = Counter(r[1] for r in results)
                amb_count = sum(v for k, v in counts.items() if k != "Clean")
                cols = st.columns(4)
                cols[0].markdown(f'<div class="metric-box"><div class="value">{len(results)}</div><div class="label">Total</div></div>', unsafe_allow_html=True)
                cols[1].markdown(f'<div class="metric-box"><div class="value">{amb_count}</div><div class="label">Ambiguous</div></div>', unsafe_allow_html=True)
                cols[2].markdown(f'<div class="metric-box"><div class="value">{counts.get("Clean",0)}</div><div class="label">Clean</div></div>', unsafe_allow_html=True)
                avg_conf = sum(r[2] for r in results) / len(results) if results else 0
                cols[3].markdown(f'<div class="metric-box"><div class="value">{avg_conf*100:.0f}%</div><div class="label">Avg Confidence</div></div>', unsafe_allow_html=True)
                st.markdown("")

            # Individual cards
            for i, (line, label, conf, flags) in enumerate(results):
                css_cls = LABEL_CSS.get(label, "clean")
                bar_clr = BAR_COLORS.get(label, "#94a3b8")
                with st.expander(f"{LABEL_EMOJI.get(label,'🔍')} [{label}]  {line[:80]}{'…' if len(line)>80 else ''}", expanded=False):
                    st.markdown(f"""
                    <div class="result-card result-{css_cls}">
                      <div class="result-label">{LABEL_EMOJI.get(label,'🔍')} {label}</div>
                      <div class="result-text">{line}</div>
                      {'<div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:'+str(conf*100)+'%;background:'+bar_clr+';"></div></div><div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;margin-top:4px;opacity:0.7;">Confidence: '+f"{conf*100:.1f}%"+'</div>' if model_loaded else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    if show_rules and flags:
                        st.markdown("**Heuristic hints:**")
                        for f in flags:
                            st.markdown(f"- {f}")


# ── Tab 3: File upload ───────────────────────
with tab_file:
    st.markdown("#### Upload an SRS document (`.txt` or `.pdf`)")
    uploaded = st.file_uploader("Choose a file", type=["txt", "pdf"])

    if uploaded is not None:
        # Extract text based on file type
        if uploaded.type == "application/pdf":
            with st.spinner("Extracting text from PDF…"):
                pdf_text = ""
                with pdfplumber.open(uploaded) as pdf:
                    for page in pdf.pages:
                        pdf_text += page.extract_text() or ""
                raw_text = pdf_text
        else:
            raw_text = uploaded.read().decode("utf-8", errors="ignore")
        
        sentences = split_sentences(raw_text)
        st.info(f"Extracted **{len(sentences)} sentences** from `{uploaded.name}`.")

        max_sentences = st.slider("Max sentences to analyze", 5, min(200, len(sentences)),
                                  min(50, len(sentences)))

        if st.button("Analyze Document", type="primary", key="btn_file"):
            to_analyze = sentences[:max_sentences]
            results = []
            prog = st.progress(0, text="Analyzing…")
            for i, sent in enumerate(to_analyze):
                if model_loaded:
                    label, conf, _ = predict_sentence(sent, tokenizer, model)
                else:
                    label, conf = "N/A", 0.0
                flags = rule_based_flags(sent) if show_rules else []
                results.append((sent, label, conf, flags))
                prog.progress((i + 1) / len(to_analyze))
            prog.empty()

            if model_loaded:
                from collections import Counter
                counts = Counter(r[1] for r in results)
                st.markdown("### 📊 Document Summary")
                cols = st.columns(len(LABEL_NAMES))
                for j, (idx, lbl) in enumerate(LABEL_NAMES.items()):
                    c = counts.get(lbl, 0)
                    cols[j].markdown(
                        f'<div class="metric-box">'
                        f'<div class="value" style="color:{BAR_COLORS[lbl]}">{c}</div>'
                        f'<div class="label">{LABEL_EMOJI[lbl]} {lbl}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown("")

            st.markdown("### 📋 Sentence-level Results")
            for sent, label, conf, flags in results:
                css_cls = LABEL_CSS.get(label, "clean")
                bar_clr = BAR_COLORS.get(label, "#94a3b8")
                if label != "Clean":   # highlight ambiguous ones expanded
                    with st.expander(f"{LABEL_EMOJI.get(label,'🔍')} [{label}]  {sent[:90]}{'…' if len(sent)>90 else ''}", expanded=True):
                        st.markdown(f"""
                        <div class="result-card result-{css_cls}">
                          <div class="result-label">{LABEL_EMOJI.get(label,'🔍')} {label}</div>
                          <div class="result-text">{sent}</div>
                          {'<div class="confidence-bar-bg"><div class="confidence-bar-fill" style="width:'+str(conf*100)+'%;background:'+bar_clr+';"></div></div><div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;margin-top:4px;opacity:0.7;">Confidence: '+f"{conf*100:.1f}%"+'</div>' if model_loaded else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        if show_rules and flags:
                            st.markdown("**Heuristic hints:**")
                            for f in flags:
                                st.markdown(f"- {f}")
                else:
                    with st.expander(f"✅ [Clean]  {sent[:90]}{'…' if len(sent)>90 else ''}", expanded=False):
                        st.markdown(f"_{sent}_")


# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94a3b8;font-size:0.8rem;'>"
    "Ambiguity Detection for SRS Documents · Fine-tuned RoBERTa · "
    "<a href='https://github.com/aashwathyadav05/Ambiguity-Detection-for-SRS-documents' "
    "style='color:#60a5fa;'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True,
)