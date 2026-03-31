# Semantic Book Recommendation System

A book recommendation system that uses **Natural Language Processing (NLP)** techniques to suggest books based on the meaning of a user's query. Built entirely with free, offline tools — no API keys required.

## How It Works

The system uses a **hybrid search architecture** combining:
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — understands semantic meaning (70% weight)
- **TF-IDF** — matches exact keywords and phrases (30% weight)

This hybrid approach outperforms either method alone, handling both semantic queries ("dark fantasy with dragons") and keyword-specific queries ("Harry Potter adventure").

## NLP Techniques Used

| # | NLP Lab Experiment | Implementation |
|---|---|---|
| 1 | Preprocessing (tokenization, stopwords, stemming, lemmatization) | `nlp_preprocessing.ipynb` |
| 2 | Bag of Words, TF-IDF, Embeddings | `feature_extraction.ipynb` |
| 3 | Morphological Analysis | `nlp_analysis.ipynb` |
| 4 | Chunking | `nlp_analysis.ipynb` |
| 5 | POS Tagging | `nlp_analysis.ipynb` |
| 6 | Named Entity Recognition | `nlp_analysis.ipynb` |
| 7 | Probability / TF-IDF Weights | `feature_extraction.ipynb` |
| 8 | Sentiment / Emotion Analysis | `sentiment_analysis.ipynb` |
| 9 | Text Classification | `text_classification.ipynb` |
| 10 | Mini Project | `gradio_dashboard.py` (full app) |

## Project Architecture

```
books_cleaned.csv
       │
       ▼
nlp_preprocessing.ipynb ──→ books_preprocessed.csv
       │
       ▼
feature_extraction.ipynb ──→ embeddings.npy + tfidf_artifacts.pkl
       │
       ▼
nlp_analysis.ipynb ──→ books_with_entities.csv
       │
       ▼
gradio_dashboard.py (Hybrid Search: Semantic 70% + TF-IDF 30%)
```

## Tech Stack

- **Python** — Core language
- **Sentence Transformers** — Semantic embeddings (all-MiniLM-L6-v2, 22M params)
- **scikit-learn** — TF-IDF, Bag of Words, Cosine Similarity
- **NLTK** — Tokenization, Stemming, Lemmatization, POS Tagging, Chunking
- **spaCy** — NER, Morphological Analysis
- **Gradio** — Web dashboard
- **Pandas / NumPy** — Data processing

## Installation

**Python:** Use **3.10+** if you want the latest spaCy releases; this repo pins **spaCy 3.8.11** so **Python 3.9** still works (3.8.13+ pulls in thinc versions that require 3.10 on Apple Silicon). Run `python -m pip install --upgrade pip` before installing if you see resolver or wheel errors.

```bash
git clone https://github.com/Shadow-code-dev/Semantic-Book-Recommendation-System.git
cd Semantic-Book-Recommendation-System

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Step 1: Run notebooks in order
1. `nlp_preprocessing.ipynb` — Preprocess book descriptions
2. `feature_extraction.ipynb` — Generate TF-IDF and embedding vectors
3. `nlp_analysis.ipynb` — POS tagging, chunking, NER analysis
4. `sentiment_analysis.ipynb` — Emotion analysis (already done)

### Step 2: Launch the dashboard
```bash
python gradio_dashboard.py
```

## Dataset

~5,200 books with metadata from [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) on Kaggle.
