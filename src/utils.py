from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re

# Expanded rule-based heuristics engine
VAGUE_WORDS = ["adequate", "appropriate", "fast", "flexible", "high", "large",
               "maximize", "minimize", "optimal", "quick", "recent", "robust",
               "simple", "small", "sufficient", "user-friendly", "various",
               "several", "many", "easy", "efficient", "modern", "normal"]
MODAL_VERBS = ["may", "might", "could", "should", "shall", "must", "will", "would", "can"]
PRONOUNS    = ["it", "they", "them", "this", "that", "these", "those", "its", "their"]

def rule_based_ambiguity(text: str) -> tuple[str, list[str]]:
    """Expanded rule-based detection. Returns primary class and a list of specific flags outlining the reasons."""
    s = text.lower()
    flags = []
    
    found_vague = [w for w in VAGUE_WORDS if re.search(rf'\b{w}\b', s)]
    if found_vague:
        flags.append(f"Vague quantifier(s): *{', '.join(found_vague)}*")
        
    found_modals = [w for w in MODAL_VERBS if re.search(rf'\b{w}\b', s)]
    if found_modals:
        weak_modals = ["may", "might", "could", "should"]
        f_weak = [w for w in found_modals if w in weak_modals]
        if f_weak:
            flags.append(f"Ambiguous/weak modal verb(s): *{', '.join(found_modals)}*")
        else:
            flags.append(f"Modal verb(s) detected: *{', '.join(found_modals)}*")
            
    found_pro = [w for w in PRONOUNS if re.search(rf'\b{w}\b', s)]
    if found_pro:
        flags.append(f"Unclear pronoun reference(s): *{', '.join(found_pro)}*")
        
    passive_re = re.compile(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', re.I)
    if passive_re.search(text):
        flags.append("Passive voice detected — actor unspecified.")
        
    attachment_re = re.compile(r'\b(with|that|which|who|where)\b.*\b(and|or)\b', re.I)
    if attachment_re.search(text):
        flags.append("Possible modifier attachment ambiguity.")
        
    mult_conj = re.search(r'\b(and|or|either|neither)\b.*\b(and|or)\b', s)
    if mult_conj:
        flags.append("Multiple conjunctions may create structural ambiguity.")

    primary_class = "Clean"
    if found_vague:
        primary_class = "Lexical"
    elif mult_conj or attachment_re.search(text):
        primary_class = "Syntactic"
    elif found_pro:
        primary_class = "Semantic"
    elif passive_re.search(text):
        primary_class = "Syntax"
    elif found_modals:
        primary_class = "Pragmatic"

    return primary_class, flags

def compute_metrics(eval_pred):
    """For Hugging Face Trainer."""
    predictions, labels = eval_pred
    pred_labels = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels, average='weighted')
    acc = accuracy_score(labels, pred_labels)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}