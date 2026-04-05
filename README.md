# Semantic Book Recommendation System

This repository is split into two self-contained projects:

| Folder | Method | Run |
|--------|--------|-----|
| **Project-semantic** | Sentence embeddings (`all-MiniLM-L6-v2`) | `cd Project-semantic && pip install -r requirements.txt && python gradio_dashboard.py` |
| **Project-tfidf** | TF-IDF + cosine similarity | `cd Project-tfidf && pip install -r requirements.txt && python gradio_dashboard.py` |

Default ports: **7860** (semantic) and **7861** (TF-IDF). Override with `GRADIO_SERVER_PORT=8080` if a port is busy.

Each folder includes the notebooks, data, and `gradio_dashboard.py` for that variant.
