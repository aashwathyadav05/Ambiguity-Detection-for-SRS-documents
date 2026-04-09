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
        rule_class, rule_flags = rule_based_ambiguity(text)

        inputs = model_obj.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model_obj.model(**inputs)
            
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
        pred_idx = int(torch.argmax(outputs.logits))
        model_conf = probs[pred_idx]
        model_pred = inverse_map.get(pred_idx, "Unknown")
        
        return {
            "text": text,
            "rule_class": rule_class,
            "rule_flags": rule_flags,
            "model_pred": model_pred,
            "model_conf": model_conf,
            "all_probs": {inverse_map.get(i, f"Label_{i}"): p for i, p in enumerate(probs)}
        }

    if args.text:
        res = predict_one(args.text)
        print(f"Text: {res['text']}")
        print(f"Rule-Based Prediction: {res['rule_class']}")
        print(f"Rule Flags: {res['rule_flags']}")
        print(f"Model-Based Prediction: {res['model_pred']} (Conf: {res['model_conf']:.2f})")
        print("Model Probability Distribution:")
        for k, v in sorted(res['all_probs'].items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:.2f}")
    elif args.file:
        df = pd.read_csv(args.file)
        results = df['text'].apply(predict_one).apply(pd.Series)
        df = pd.concat([df, results.drop('text', axis=1)], axis=1)
        df.to_csv("predictions.csv", index=False)
        print(f"Predictions saved to predictions.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()