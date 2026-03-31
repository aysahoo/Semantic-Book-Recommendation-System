# Semantic Book Recommendation System
### NLP Lab Mini Project (Experiment 10)

A content-based book recommendation system built using core **Natural Language Processing** techniques. The system processes 5,197 book descriptions through a complete NLP pipeline — from text preprocessing to semantic search — and presents recommendations through an interactive web dashboard.

---

## Problem Statement

Readers face difficulty discovering books that match their interests. Traditional search relies on exact keyword matching, which fails when users describe what they want in natural language (e.g., "a dark fantasy story with dragons"). This project applies NLP techniques to understand the **meaning** behind user queries and recommend books with semantically similar descriptions.

---

## NLP Techniques Implemented

| Exp # | NLP Lab Experiment | Technique Used | File |
|-------|-------------------|---------------|------|
| 1 | Text Preprocessing | Tokenization, Stop Word Removal, Stemming, Lemmatization | `nlp_preprocessing.ipynb` |
| 2 | Feature Extraction | Bag of Words, TF-IDF Vectorization, Sentence Embeddings | `feature_extraction.ipynb` |
| 3 | Morphological Analysis | Prefix/Suffix/Root analysis, Lemma extraction using spaCy | `nlp_analysis.ipynb` |
| 4 | Chunking | Noun Phrase chunking using NLTK RegexpParser | `nlp_analysis.ipynb` |
| 5 | POS Tagging | Part-of-Speech tagging using NLTK and spaCy | `nlp_analysis.ipynb` |
| 6 | Named Entity Recognition | Person, Location, Organization extraction using spaCy | `nlp_analysis.ipynb` |
| 7 | Probability & Word Importance | TF-IDF weights as probabilistic word importance scores | `feature_extraction.ipynb` |
| 8 | Sentiment Analysis | Emotion detection (joy, sadness, anger, fear, surprise, disgust) | `sentiment_analysis.ipynb` |
| 9 | Text Classification | Category classification of book descriptions | `data_preprocessing.ipynb` |
| 10 | Mini Project | Complete recommendation system with Gradio dashboard | `gradio_dashboard.py` |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                     │
│  books_cleaned.csv (5197 books from Kaggle)                        │
│       │                                                             │
│       ├──► nlp_preprocessing.ipynb                                  │
│       │      Tokenize → Remove Stopwords → Lemmatize               │
│       │      Output: books_preprocessed.csv                        │
│       │                                                             │
│       ├──► feature_extraction.ipynb                                  │
│       │      Bag of Words → TF-IDF Matrix → Sentence Embeddings    │
│       │      Output: tfidf_artifacts.pkl, embeddings.npy           │
│       │                                                             │
│       ├──► sentiment_analysis.ipynb                                  │
│       │      Emotion scoring using DistilRoBERTa transformer       │
│       │      Output: books_with_emotions.csv                       │
│       │                                                             │
│       └──► nlp_analysis.ipynb                                       │
│              POS Tagging → Chunking → Morphology → NER             │
│              Output: books_with_entities.csv                       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                     RECOMMENDATION ENGINE                           │
│                                                                     │
│  User Query: "dark fantasy with dragons"                           │
│       │                                                             │
│       ├──► Sentence Transformer (all-MiniLM-L6-v2)                 │
│       │      Encodes query into 384-dim semantic vector             │
│       │      Cosine similarity with all book embeddings            │
│       │      Weight: 70%                                           │
│       │                                                             │
│       ├──► TF-IDF Vectorizer                                        │
│       │      Transforms query using fitted vocabulary              │
│       │      Cosine similarity with TF-IDF matrix                  │
│       │      Weight: 30%                                           │
│       │                                                             │
│       └──► Hybrid Score = 0.7 × Semantic + 0.3 × Keyword           │
│              Filter by: Category + Emotional Tone                  │
│              Return top 8 recommendations                          │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        GRADIO DASHBOARD                             │
│                                                                     │
│  Tab 1: Search Books — Hybrid recommendation engine                │
│  Tab 2: Browse All Books — Paginated view with all NLP data        │
│  Tab 3: NLP Analysis Demo — Live POS/Chunking/Morphology/NER      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How Each NLP Technique Is Applied

### 1. Text Preprocessing (`nlp_preprocessing.ipynb`)
- **Tokenization**: Splits book descriptions into individual words using NLTK `word_tokenize`
- **Stop Word Removal**: Filters out common English words (the, is, and, a) that don't carry meaning
- **Stemming**: Reduces words to root form using Porter Stemmer (running → run, stories → stori)
- **Lemmatization**: Converts words to dictionary form using WordNet (running → run, better → good)
- **Output**: Cleaned descriptions stored in `books_preprocessed.csv`

### 2. Feature Extraction (`feature_extraction.ipynb`)
- **Bag of Words**: Counts word occurrences per book using `CountVectorizer` (5000 features)
- **TF-IDF**: Calculates word importance using Term Frequency-Inverse Document Frequency with bigrams
- **Sentence Embeddings**: Generates 384-dimensional dense vectors using `all-MiniLM-L6-v2` transformer model
- **Output**: `tfidf_artifacts.pkl` (vectorizer + matrix), `embeddings.npy` (sentence embeddings)

### 3. Morphological Analysis (`nlp_analysis.ipynb`)
- Analyzes word structure: prefix, suffix, root, lemma
- Example: "unbelievable" → prefix: un-, root: believe, suffix: -able
- Uses spaCy's morphological features for detailed analysis

### 4. Chunking (`nlp_analysis.ipynb`)
- Extracts noun phrases from descriptions using NLTK `RegexpParser`
- Grammar rule: `NP: {<DT|PP$>?<JJ>*<NN.*>+}`
- Example: "the angry visionary", "abolitionist cause", "young son"

### 5. POS Tagging (`nlp_analysis.ipynb`)
- Tags each word with its grammatical role (Noun, Verb, Adjective, etc.)
- Uses both NLTK `pos_tag` and spaCy for comparison
- Example: John(NOUN) went(VERB) to(PREP) New(ADJ) York(NOUN)

### 6. Named Entity Recognition (`nlp_analysis.ipynb`)
- Extracts real-world entities from all 5,197 book descriptions using spaCy
- Categories: PERSON (character names), GPE/LOC (places), ORG (organizations)
- Results: 3,717 books with person data, 2,404 with location data, 2,638 with organization data
- **Output**: `books_with_entities.csv`

### 7. Sentiment/Emotion Analysis (`sentiment_analysis.ipynb`)
- Uses `j-hartmann/emotion-english-distilroberta-base` transformer model
- Scores each description for 6 emotions: joy, sadness, anger, fear, surprise, disgust
- Enables emotion-based filtering in the dashboard (e.g., show only "happy" books)
- **Output**: `books_with_emotions.csv`

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Preprocessing | NLTK | Tokenization, stemming, lemmatization, POS tagging, chunking |
| NER & Morphology | spaCy | Named entity recognition, morphological analysis |
| Feature Extraction | scikit-learn | Bag of Words, TF-IDF, cosine similarity |
| Semantic Search | sentence-transformers | all-MiniLM-L6-v2 (22M params, 384-dim embeddings) |
| Emotion Analysis | transformers (HuggingFace) | DistilRoBERTa emotion classifier |
| Dashboard | Gradio | Interactive web interface |
| Data Processing | Pandas, NumPy | DataFrame operations, matrix storage |

---

## Dataset

- **Source**: [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) (Kaggle)
- **Size**: 5,197 books after cleaning
- **Fields**: Title, Authors, Description, Categories, Rating, Pages, Thumbnail URL

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Shadow-code-dev/Semantic-Book-Recommendation-System.git
cd Semantic-Book-Recommendation-System

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the Project

### Step 1: Run the notebooks in order
```
1. data_preprocessing.ipynb       → Cleans raw data → books_cleaned.csv
2. nlp_preprocessing.ipynb        → Preprocesses text → books_preprocessed.csv
3. feature_extraction.ipynb       → Extracts features → embeddings.npy, tfidf_artifacts.pkl
4. sentiment_analysis.ipynb       → Emotion scoring → books_with_emotions.csv
5. nlp_analysis.ipynb             → NER extraction → books_with_entities.csv
```

### Step 2: Launch the dashboard
```bash
python gradio_dashboard.py
```
Opens at `http://127.0.0.1:7860`

---

## Project Files

```
Semantic-Book-Recommendation-System/
├── data_preprocessing.ipynb      # Data cleaning and preparation
├── nlp_preprocessing.ipynb       # Exp 1: Tokenization, stopwords, stemming, lemmatization
├── feature_extraction.ipynb      # Exp 2, 7: BoW, TF-IDF, sentence embeddings
├── sentiment_analysis.ipynb      # Exp 8: Emotion analysis
├── nlp_analysis.ipynb            # Exp 3, 4, 5, 6: Morphology, chunking, POS, NER
├── gradio_dashboard.py           # Exp 10: Complete recommendation dashboard
├── requirements.txt              # Python dependencies
├── book_not_found.jpg            # Fallback thumbnail
└── README.md                     # Project documentation
```

---

## Results

- Hybrid search (Semantic 70% + TF-IDF 30%) provides accurate recommendations for both natural language queries and keyword-specific searches
- Emotion-based filtering allows users to find books matching a specific mood
- NER extraction identified entities across 3,717+ books
- The system runs fully offline with no API dependencies
