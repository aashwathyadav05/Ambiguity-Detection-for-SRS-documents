from transformers import AutoModelForSequenceClassification, AutoConfig
from .preprocessor import get_tokenizer

LABEL_MAP = {
    "Lexical ambiguity": 0,
    "Syntactic ambiguity": 1,
    "Semantic ambiguity": 2,
    "Syntax ambiguity": 3,
    "Pragmatic ambiguity": 4,
    "Clean": 5,
    # Handle possible variations in dataset
    "lexical": 0, "syntactic": 1, "semantic": 2, "syntax": 3, "pragmatic": 4, "clean": 5,
}

class AmbiguityRoBERTa:
    def __init__(self, model_name: str = "roberta-base", num_labels: int = 6):
        self.tokenizer = get_tokenizer(model_name)
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f" Model saved to {path}")