# Ambiguity Detection in SRS Documents using RoBERTa

**NLP-based tool for automatically detecting and classifying ambiguities** in Software Requirements Specification (SRS) documents.  
Fine-tuned **RoBERTa** model identifies linguistic ambiguities that can lead to faults in software development.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co/docs/transformers/index)
[![React](https://img.shields.io/badge/React-19-blue)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
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

## Architecture

| Layer | Technology | Details |
|-------|-----------|---------|
| Frontend | React + Vite | Modern UI with drag-and-drop file upload, interactive result cards |
| Backend | FastAPI | REST API serving the RoBERTa model + rule-based engine |
| ML Model | RoBERTa (HuggingFace) | Fine-tuned on SRS ambiguity dataset, 6-class classification |

## Project Structure

```text
Ambiguity-Detection-for-SRS-documents/
│
├── backend/                   # FastAPI backend
│   ├── main.py                # API endpoints (health, labels, analyze)
│   └── requirements.txt       # Backend Python dependencies
│
├── frontend/                  # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx            # Main application
│   │   ├── components/        # Hero, LabelGuide, FileUpload, etc.
│   │   └── api/client.js      # API client
│   ├── index.html
│   └── package.json
│
├── src/                       # ML modules (shared by backend & scripts)
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model.py
│   └── utils.py
│
├── scripts/
│   ├── train.py               # CLI training
│   └── predict.py             # CLI inference
│
├── models/                    # Trained model weights (git-ignored)
│   └── roberta-ambiguity-final/
│
├── data/
│   └── dataset.csv            # Training dataset
│
├── notebooks/
│   └── 01_exploration_and_training.ipynb
│
├── requirements.txt           # Training/notebook dependencies
├── app.py                     # Legacy Streamlit app (reference only)
└── README.md
```

## Quick Start

### 1. Backend

```bash
# Install backend dependencies (from project root)
pip install -r backend/requirements.txt

# Start the FastAPI server
uvicorn backend.main:app --reload
# Server runs at http://localhost:8000
```

### 2. Frontend

```bash
# Install frontend dependencies
cd frontend
npm install

# Start the dev server
npm run dev
# App runs at http://localhost:5173
```

### 3. Use the app

1. Open `http://localhost:5173` in your browser
2. Upload a `.txt` or `.pdf` SRS document
3. Adjust the max sentences slider
4. Click **Analyze Document**
5. View sentence-level results with model predictions, confidence scores, and rule-based flags

## Model Choice: Why RoBERTa?

We selected **RoBERTa-base** because:
- Optimized pre-training (longer training, dynamic masking, larger batches) → better performance than BERT on many NLP classification tasks
- Strong contextual understanding → effective for detecting syntactic, semantic, and pragmatic ambiguities in technical text
- Proven in similar requirements engineering / NLP4RE tasks (ambiguity, anaphora, defect detection)
- Efficient fine-tuning on modest hardware (e.g., Colab GPU)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check + model status |
| `/api/labels` | GET | Label metadata (names, descriptions, colors) |
| `/api/analyze` | POST | Upload & analyze a document (multipart form) |

## License

MIT
