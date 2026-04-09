import re
from transformers import AutoTokenizer

def clean_text(text: str) -> str:
    """Basic cleaning (used for rule-based heuristics). The underlying RoBERTa tokenizer handles most of this."""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?]', '', text)
    return text.lower()

def get_tokenizer(model_name: str = "roberta-base"):
    # Uses AutoTokenizer (best practice) but specifically loads the RoBERTa tokenizer dynamically
    return AutoTokenizer.from_pretrained(model_name)