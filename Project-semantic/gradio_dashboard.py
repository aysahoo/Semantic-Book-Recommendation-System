import os
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import gradio as gr

nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

nlp = spacy.load("en_core_web_sm")

books = pd.read_csv('books_with_emotions.csv')
books_entities = pd.read_csv('books_with_entities.csv')

entity_cols = books_entities[["isbn13", "persons", "locations", "organizations"]].copy()
books_preprocessed = pd.read_csv('books_preprocessed.csv')[["isbn13", "processed_description"]]
books = books.merge(entity_cols, on="isbn13", how="left")
books = books.merge(books_preprocessed, on="isbn13", how="left")
books[["persons", "locations", "organizations"]] = books[["persons", "locations", "organizations"]].fillna("")
books["processed_description"] = books["processed_description"].fillna("")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(),
                                    "book_not_found.jpg",
                                    books["large_thumbnail"])

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = np.load("embeddings.npy")


# --- Recommendation Logic (sentence embeddings only) ---

def retrieve_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 8
) -> pd.DataFrame:
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = scores.argsort()[-initial_top_k:][::-1]
    books_recs = books.iloc[top_indices].head(final_top_k).copy()

    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return books_recs


def recommend_books(query, category, tone):
    if not query or not query.strip():
        return []
    recommendations = retrieve_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


# --- Browse Books Logic ---

BOOKS_PER_PAGE = 10

def get_browse_data(category, page):
    page = int(page)
    filtered = books if category == "All" else books[books["simple_categories"] == category]
    total_pages = max(1, (len(filtered) + BOOKS_PER_PAGE - 1) // BOOKS_PER_PAGE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * BOOKS_PER_PAGE
    end = start + BOOKS_PER_PAGE
    page_books = filtered.iloc[start:end]

    html = f"""<p style='text-align:center; color:#222; font-weight:bold;'>Showing {start+1}–{min(end, len(filtered))} of {len(filtered)} books  |  Page {page} of {total_pages}</p>"""

    for i, (_, row) in enumerate(page_books.iterrows()):
        thumb = row.get("large_thumbnail", "book_not_found.jpg")
        title = row.get("title", "Unknown")
        authors = str(row.get("authors", "Unknown")).replace(";", ", ")
        cat = row.get("simple_categories", "")
        year = int(row["published_year"]) if pd.notna(row.get("published_year")) else ""
        rating = row.get("average_rating", "")
        pages = int(row["num_pages"]) if pd.notna(row.get("num_pages")) else ""
        desc = str(row.get("description", ""))[:250]
        proc_desc = str(row.get("processed_description", ""))[:250]

        joy = row.get("joy", 0)
        sadness = row.get("sadness", 0)
        anger = row.get("anger", 0)
        fear = row.get("fear", 0)
        surprise = row.get("surprise", 0)
        disgust = row.get("disgust", 0)
        top_emotion = max(
            [("Joy", joy), ("Sadness", sadness), ("Anger", anger),
             ("Fear", fear), ("Surprise", surprise), ("Disgust", disgust)],
            key=lambda x: x[1]
        )

        persons = str(row.get("persons", ""))
        locations = str(row.get("locations", ""))
        orgs = str(row.get("organizations", ""))

        ner_parts = []
        if persons and persons != "nan":
            ner_parts.append(f"<b>Persons:</b> {persons}")
        if locations and locations != "nan":
            ner_parts.append(f"<b>Locations:</b> {locations}")
        if orgs and orgs != "nan":
            ner_parts.append(f"<b>Orgs:</b> {orgs}")
        ner_text = "<br>".join(ner_parts) if ner_parts else "—"

        bg = "#1a1a2e" if i % 2 == 0 else "#16213e"

        html += f"""
        <div style="background:{bg}; border:1px solid #334155; border-radius:8px; margin:10px 0; padding:16px; color:#e2e8f0;">
            <div style="display:flex; gap:16px;">
                <img src="{thumb}" style="width:70px; height:105px; object-fit:contain; border-radius:4px; background:#0f0f23; flex-shrink:0;" />
                <div style="flex:1;">
                    <h3 style="margin:0 0 4px 0; color:#f1f5f9;">{title}</h3>
                    <p style="margin:0; color:#94a3b8;"><b style="color:#cbd5e1;">{authors}</b> | {cat} | {year} | Rating: {rating} | {pages} pages</p>
                </div>
            </div>

            <table style="width:100%; border-collapse:collapse; margin-top:12px; font-size:0.85em;">
                <tr style="background:#0f172a; color:#38bdf8;">
                    <th style="padding:8px; text-align:left; width:50%;">Original Description</th>
                    <th style="padding:8px; text-align:left; width:50%;">Preprocessed Description (After NLP Pipeline)</th>
                </tr>
                <tr>
                    <td style="padding:8px; color:#cbd5e1; vertical-align:top; border:1px solid #334155; background:#1e293b;">{desc}...</td>
                    <td style="padding:8px; color:#cbd5e1; vertical-align:top; border:1px solid #334155; background:#1e293b;">{proc_desc}...</td>
                </tr>
            </table>

            <table style="width:100%; border-collapse:collapse; margin-top:8px; font-size:0.85em;">
                <tr style="background:#0f172a; color:#38bdf8;">
                    <th style="padding:8px; text-align:left;">Dominant Emotion</th>
                    <th style="padding:8px; text-align:left;">Joy</th>
                    <th style="padding:8px; text-align:left;">Sadness</th>
                    <th style="padding:8px; text-align:left;">Anger</th>
                    <th style="padding:8px; text-align:left;">Fear</th>
                    <th style="padding:8px; text-align:left;">Surprise</th>
                    <th style="padding:8px; text-align:left;">Disgust</th>
                </tr>
                <tr style="background:#1e293b;">
                    <td style="padding:8px; color:#fbbf24; font-weight:bold; border:1px solid #334155;">{top_emotion[0]} ({top_emotion[1]:.2f})</td>
                    <td style="padding:8px; color:#e2e8f0; border:1px solid #334155;">{joy:.3f}</td>
                    <td style="padding:8px; color:#e2e8f0; border:1px solid #334155;">{sadness:.3f}</td>
                    <td style="padding:8px; color:#e2e8f0; border:1px solid #334155;">{anger:.3f}</td>
                    <td style="padding:8px; color:#e2e8f0; border:1px solid #334155;">{fear:.3f}</td>
                    <td style="padding:8px; color:#e2e8f0; border:1px solid #334155;">{surprise:.3f}</td>
                    <td style="padding:8px; color:#e2e8f0; border:1px solid #334155;">{disgust:.3f}</td>
                </tr>
            </table>

            <table style="width:100%; border-collapse:collapse; margin-top:8px; font-size:0.85em;">
                <tr style="background:#0f172a; color:#38bdf8;">
                    <th style="padding:8px; text-align:left;">NER — Extracted Entities</th>
                </tr>
                <tr style="background:#1e293b;">
                    <td style="padding:8px; color:#e2e8f0; border:1px solid #334155;">{ner_text}</td>
                </tr>
            </table>
        </div>"""

    return html, str(page)


def next_page(category, page):
    return get_browse_data(category, int(page) + 1)

def prev_page(category, page):
    return get_browse_data(category, max(1, int(page) - 1))

def reset_page(category):
    return get_browse_data(category, 1)


# --- NLP Demo Logic ---

def analyze_text(text):
    if not text or not text.strip():
        return "Please enter some text to analyze."

    tokens = word_tokenize(text)
    pos_tags_list = pos_tag(tokens)

    pos_rows = "".join(f"<tr><td>{w}</td><td><code>{t}</code></td></tr>" for w, t in pos_tags_list[:30])
    pos_html = f"""
    <h3>🏷️ POS Tagging</h3>
    <table style="border-collapse:collapse; width:100%;">
        <tr style="background:#e2e8f0;"><th style="padding:6px; text-align:left;">Token</th><th style="padding:6px; text-align:left;">POS Tag</th></tr>
        {pos_rows}
    </table>
    """

    grammar = r"NP: {<DT|PP\$>?<JJ>*<NN.*>+}"
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags_list)
    noun_phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        noun_phrases.append(" ".join(w for w, _ in subtree.leaves()))
    chunks_html = "<h3>🧩 Chunking (Noun Phrases)</h3><ul>"
    chunks_html += "".join(f"<li>{np}</li>" for np in noun_phrases) if noun_phrases else "<li><i>No noun phrases found</i></li>"
    chunks_html += "</ul>"

    doc = nlp(text)
    morph_rows = "".join(
        f"<tr><td>{t.text}</td><td>{t.lemma_}</td><td><code>{t.pos_}</code></td><td>{t.morph}</td></tr>"
        for t in doc[:25]
    )
    morph_html = f"""
    <h3>🔬 Morphological Analysis</h3>
    <table style="border-collapse:collapse; width:100%;">
        <tr style="background:#e2e8f0;"><th style="padding:6px;">Token</th><th style="padding:6px;">Lemma</th><th style="padding:6px;">POS</th><th style="padding:6px;">Morphology</th></tr>
        {morph_rows}
    </table>
    """

    ner_rows = "".join(
        f"<tr><td>{ent.text}</td><td><code>{ent.label_}</code></td><td>{spacy.explain(ent.label_)}</td></tr>"
        for ent in doc.ents
    )
    ner_html = f"""
    <h3>🏷️ Named Entity Recognition</h3>
    <table style="border-collapse:collapse; width:100%;">
        <tr style="background:#e2e8f0;"><th style="padding:6px;">Entity</th><th style="padding:6px;">Label</th><th style="padding:6px;">Description</th></tr>
        {ner_rows if ner_rows else '<tr><td colspan="3"><i>No entities found</i></td></tr>'}
    </table>
    """

    return pos_html + chunks_html + morph_html + ner_html


# --- UI ---

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

css = """
.book-gallery .gallery-item > div { aspect-ratio: 2/3 !important; }
.book-gallery .gallery-item img { object-fit: contain !important; border-radius: 6px; background: #f5f5f5; }
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as dashboard:
    gr.Markdown("# Semantic Book Recommendation System (sentence embeddings)")

    with gr.Tabs():

        # --- Tab 1: Search ---
        with gr.TabItem("Search Books"):
            with gr.Row():
                user_query = gr.Textbox(label="Please enter a description of a book:",
                                        placeholder="e.g. A story about love and adventure")
                category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
                tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
                submit_button = gr.Button("Search Book Library")

            gr.Markdown("### Found Recommendations")
            output = gr.Gallery(
                label="Recommended books",
                columns=4,
                rows=2,
                height=650,
                object_fit="contain",
                elem_classes=["book-gallery"]
            )

            submit_button.click(
                fn=recommend_books,
                inputs=[user_query, category_dropdown, tone_dropdown],
                outputs=output
            )

        # --- Tab 2: Browse All Books ---
        with gr.TabItem("Browse All Books"):
            gr.Markdown("### Browse the complete book library with all extracted NLP data")
            with gr.Row():
                browse_category = gr.Dropdown(choices=categories, label="Filter by category:", value="All")
                page_num = gr.Textbox(label="Page", value="1", scale=0)
            with gr.Row():
                prev_btn = gr.Button("← Previous")
                next_btn = gr.Button("Next →")

            browse_output = gr.HTML()

            browse_category.change(fn=reset_page, inputs=[browse_category], outputs=[browse_output, page_num])
            next_btn.click(fn=next_page, inputs=[browse_category, page_num], outputs=[browse_output, page_num])
            prev_btn.click(fn=prev_page, inputs=[browse_category, page_num], outputs=[browse_output, page_num])

            dashboard.load(fn=get_browse_data, inputs=[browse_category, page_num], outputs=[browse_output, page_num])

        # --- Tab 3: NLP Demo ---
        with gr.TabItem("NLP Analysis Demo"):
            gr.Markdown("### Enter any text to see POS Tagging, Chunking, Morphology & NER in action")
            nlp_input = gr.Textbox(
                label="Enter text to analyze:",
                placeholder="e.g. John went to New York to visit the Empire State Building.",
                lines=3
            )
            nlp_button = gr.Button("Analyze")
            nlp_output = gr.HTML()

            nlp_button.click(fn=analyze_text, inputs=[nlp_input], outputs=[nlp_output])

if __name__ == "__main__":
    _port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    dashboard.launch(show_api=False, server_port=_port)
