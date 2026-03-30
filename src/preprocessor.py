import re
from transformers import RobertaTokenizer

def clean_text(text: str) -> str:
    """Basic cleaning (used for rule-based heuristics). RoBERTa tokenizer handles most of this."""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?]', '', text)
    return text.lower()

def get_tokenizer(model_name: str = "roberta-base"):
    return RobertaTokenizer.from_pretrained(model_name)