# scripts/train.py
import argparse
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_data
from src.model import AmbiguityRoBERTa, LABEL_MAP
from src.utils import compute_metrics

from datasets import Dataset
from transformers import Trainer, TrainingArguments

def main():
    parser = argparse.ArgumentParser(description="Train RoBERTa for SRS Ambiguity Detection")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_dir", type=str, default="models/roberta-ambiguity")
    args = parser.parse_args()

    print("Starting Ambiguity Detection Training...\n")

    # Load data - now looks for data/dataset.csv
    df = load_data("data/dataset.csv")

    # Map labels to integers
    df['label_id'] = df['label'].astype(str).str.strip().map(LABEL_MAP)
    df = df.dropna(subset=['label_id']).reset_index(drop=True)

    print(f"Loaded {len(df)} valid samples | Classes: {df['label'].nunique()}")

    # Train/Val split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']].rename(columns={'label_id': 'label'}))
    val_dataset = Dataset.from_pandas(val_df[['text', 'label_id']].rename(columns={'label_id': 'label'}))

    # Load model and tokenizer
    model_obj = AmbiguityRoBERTa()

    def tokenize_fn(examples):
        return model_obj.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="logs",
        logging_steps=50,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model_obj.model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=model_obj.tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"Starting training for {args.epochs} epochs...")
    trainer.train()

    # Save final model
    model_obj.save(args.model_dir)
    print(f"\nTraining completed! Model saved to: {args.model_dir}")


if __name__ == "__main__":
    main()