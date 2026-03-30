from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re

# Simple rule-based heuristics (for interpretability + hybrid fallback)
def rule_based_ambiguity(text: str) -> str:
    """Quick rule-based detection. Returns class or 'Unknown'."""
    text = text.lower()
    if any(word in text for word in ["may", "might", "could", "should", "can"]):
        return "Pragmatic ambiguity"
    if re.search(r'\b(and|or|either|neither)\b.*\b(and|or)\b', text):
        return "Syntactic ambiguity"
    if any(ambig in text for ambig in ["some", "few", "many", "several", "a number of"]):
        return "Lexical ambiguity"
    # Add more rules as needed
    return "Clean"  # fallback

def compute_metrics(eval_pred):
    """For Hugging Face Trainer."""
    predictions, labels = eval_pred
    pred_labels = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels, average='weighted')
    acc = accuracy_score(labels, pred_labels)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}