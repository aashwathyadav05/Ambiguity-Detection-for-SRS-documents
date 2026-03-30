# scripts/predict.py
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import AmbiguityRoBERTa, LABEL_MAP
from src.utils import rule_based_ambiguity

def main():
    parser = argparse.ArgumentParser(description="Predict ambiguity in SRS requirements")
    parser.add_argument("--text", type=str, help="Single requirement text")
    parser.add_argument("--file", type=str, help="CSV file with 'text' column")
    parser.add_argument("--model_path", type=str, default="models/roberta-ambiguity-final")
    args = parser.parse_args()

    # Load model
    model_obj = AmbiguityRoBERTa()
    model_obj.model = model_obj.model.from_pretrained(args.model_path)
    model_obj.tokenizer = model_obj.tokenizer.from_pretrained(args.model_path)
    model_obj.model.eval()

    inverse_map = {v: k for k, v in LABEL_MAP.items()}

    def predict_one(text: str):
        rule = rule_based_ambiguity(text)
        if rule != "Clean":
            return rule

        inputs = model_obj.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model_obj.model(**inputs)
        pred = outputs.logits.argmax(-1).item()
        return inverse_map.get(pred, "Unknown")

    if args.text:
        result = predict_one(args.text)
        print(f"Text: {args.text}")
        print(f"Predicted: {result}")
    elif args.file:
        df = pd.read_csv(args.file)
        df['predicted'] = df['text'].apply(predict_one)
        df.to_csv("predictions.csv", index=False)
        print(f"Predictions saved to predictions.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()