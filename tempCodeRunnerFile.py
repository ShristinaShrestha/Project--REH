from flask import Flask, render_template, request, redirect, url_for
import os
import re
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Folder where the documents are stored
DOCUMENTS_FOLDER = "static/dataset_final"  # Ensure this folder exists

# Load necessary NLTK resources
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load documents from folder
def load_documents():
    documents = {}
    doc_id_to_filename = {}
    try:
        for i, filename in enumerate(os.listdir(DOCUMENTS_FOLDER)):
            if filename.endswith(".txt"):
                with open(os.path.join(DOCUMENTS_FOLDER, filename), "r", encoding="utf-8") as f:
                    content = f.read()
                    documents[i] = content
                    doc_id_to_filename[i] = filename
    except FileNotFoundError:
        print(f"Error: Directory '{DOCUMENTS_FOLDER}' not found.")
        raise
    return documents, doc_id_to_filename

# Preprocess text: lowercase, remove special characters, remove stopwords, lemmatize
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters and punctuation
    tokens = word_tokenize(text)  # Tokenize the text
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return cleaned_tokens

# Build vocabulary (unique terms across all documents)
def build_vocab(documents):
    vocab = set()
    for doc in documents.values():
        vocab.update(clean_text(doc))
    return sorted(vocab)

# Calculate term frequency (TF)
def term_frequency(term, document):
    return document.count(term) / len(document)

# Calculate inverse document frequency (IDF)
def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (1 + num_docs_containing_term))

# Compute TF-IDF vector for a document
def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:
        tf = term_frequency(term, document)
        idf = inverse_document_frequency(term, all_documents)
        tfidf_vector.append(tf * idf)
    return np.array(tfidf_vector)

# Calculate TF-IDF vectors for all documents
def calculate_tfidf_vectors(documents, vocab):
    tokenized_docs = {i: clean_text(doc) for i, doc in documents.items()}
    doc_tfidf_vectors = {i: compute_tfidf(doc, tokenized_docs.values(), vocab) for i, doc in tokenized_docs.items()}
    return doc_tfidf_vectors

# Load documents and prepare the TF-IDF vectors
documents, doc_id_to_filename = load_documents()
vocab = build_vocab(documents)
doc_tfidf_vectors = calculate_tfidf_vectors(documents, vocab)

# Search functionality
@app.route("/", methods=["GET", "POST"])
def search():
    results = None
    query = None
    message = None

    if request.method == "POST":
        query = request.form["query"]  # Get the user's query
        tokenized_query = clean_text(query)  # Preprocess the query
        query_vector = compute_tfidf(tokenized_query, documents.values(), vocab)  # Compute TF-IDF for the query

        # Calculate cosine similarity between query and all document vectors
        similarities = {i: cosine_similarity([query_vector], [doc_vector])[0][0] for i, doc_vector in doc_tfidf_vectors.items()}

        # Sort documents by similarity score
        ranked_docs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

        # Filter documents with similarity > 0
        top_docs_with_scores = [(doc_id_to_filename[doc_id].replace(".txt", ""), score) for doc_id, score in ranked_docs if score > 0]

        if not top_docs_with_scores:
            message = "No results matched your search."
        else:
            results = top_docs_with_scores[:5]  # Limit to top 5 results

    return render_template("index.html", query=query, docs=results, message=message)

# Article route to display the document content
@app.route("/article/<filename>")
def article(filename):
    # Replace underscores with spaces when accessing the file
    filename = filename.replace('_', ' ')
    filepath = os.path.join(DOCUMENTS_FOLDER, f"{filename}.txt")
    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            content = f.read()
        return render_template("article.html", filename=filename, content=content)
    except FileNotFoundError:
        return "Document not found", 404

if __name__ == "__main__":
    app.run(debug=True)
