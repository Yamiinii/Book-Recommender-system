import numpy as np
import pandas as pd
from fastapi import FastAPI
import gradio as gr
import uvicorn

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize FastAPI app
app = FastAPI()

# Load book dataset
books = pd.read_csv('books_with_emotions.csv')
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "cover-not-found.jpg", books["large_thumbnail"])

# Load documents for vector search
raw_document = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_document)

# Initialize embedding model and vector database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vector_db = Chroma.from_documents(documents, embedding_model)


# Function to retrieve recommendations
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    similar_docs = vector_db.similarity_search(query, k=50)
    matched_isbn_list = [
        int(doc.page_content.strip('"').split()[0]) for doc in similar_docs
    ]

    books_recs = books[books["isbn13"].isin(matched_isbn_list)].head(initial_top_k)

    if category and category != "All":
        books_recs = books_recs.loc[books_recs["simple_categories"] == category].head(final_top_k)
    else:
        books_recs = books_recs.head(final_top_k)

    if tone:
        emotion_mapping = {
            "Happy": "joy",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness"
        }
        if tone in emotion_mapping:
            books_recs = books_recs.sort_values(by=emotion_mapping[tone], ascending=False)

    return books_recs


# Gradio recommendation function
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


# Define categories and tones
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Define Gradio app inside FastAPI
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

        gr.Markdown("# Recommendations")
        output = gr.Gallery(label="Recommended books", columns=8, rows=2)

        submit_button.click(fn=recommend_books,
                            inputs=[user_query, category_dropdown, tone_dropdown],
                            outputs=output)


# Expose the Gradio app through FastAPI
@app.get("/")
def home():
    return {"message": "Gradio Book Recommender Running!"}


@app.get("/gradio")
def gradio_interface():
    return dashboard.launch(share=True, server_name="0.0.0.0", server_port=8000)


# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
