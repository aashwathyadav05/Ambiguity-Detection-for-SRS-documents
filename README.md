# Ambiguity Detection in SRS Documents using RoBERTa

**NLP-based tool for automatically detecting and classifying ambiguities** in Software Requirements Specification (SRS) documents.  
Fine-tuned **RoBERTa** model identifies linguistic ambiguities that can lead to faults in software development.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Software Requirements Specifications (SRS) often contain ambiguous language (e.g., vague quantifiers, unclear pronouns, modal verbs, attachment ambiguities), leading to misunderstandings, rework, and project failures.  

This project builds an **ambiguity detection system** that:
- Processes SRS text (individual requirements or full documents)
- Classifies each requirement sentence into one of **6 categories**:
  1. Lexical ambiguity
  2. Syntactic ambiguity
  3. Semantic ambiguity
  4. Syntax ambiguity
  5. Pragmatic ambiguity
  6. Clean (unambiguous)
- Combines **rule-based heuristics** (for interpretability) with a fine-tuned **RoBERTa** model (for high accuracy on complex cases)

**Primary Dataset**: [Fault-prone SRS Dataset](https://www.kaggle.com/datasets/corpus4panwo/fault-prone-srs-dataset) (~7,061 labeled requirements from ~200 publicly collected SRS documents, as described in the 2023 MDPI paper).

## Model Choice: Why RoBERTa?

We selected **RoBERTa-base** because:
- Optimized pre-training (longer training, dynamic masking, larger batches) → better performance than BERT on many NLP classification tasks
- Strong contextual understanding → effective for detecting syntactic, semantic, and pragmatic ambiguities in technical text
- Proven in similar requirements engineering / NLP4RE tasks (ambiguity, anaphora, defect detection)
- Efficient fine-tuning on modest hardware (e.g., Colab GPU)

## Project Structure
```text
ambiguity-detection-srs-roberta/
│
├── data/                      # Download dataset here (do NOT commit large files)
│   └── raw/
│       └── dataset.csv        # Place the Kaggle file here
│
├── notebooks/
│   └── 01_exploration_and_training.ipynb   # Main notebook (start here)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model.py
│   └── utils.py
│
├── scripts/
│   ├── train.py               # CLI training
│   └── predict.py             # Inference on new text
│
├── models/                    # Where trained model will be saved
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE                    # MIT
```
