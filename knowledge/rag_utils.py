# knowledge/rag_utils.py

"""
This module provides utilities for the Retrieval-Augmented Generation (RAG) component of the pipeline.
Its main responsibilities are:

1.  **PDF Text Extraction**: Reads and extracts all text content from a given PDF file.
    It uses a simple in-memory cache to avoid re-reading the same file multiple times.

2.  **Text Chunking**: Splits the extracted text into smaller, manageable chunks. This is crucial
    for creating effective embeddings for semantic search.

3.  **Embedding Generation**: Uses a sentence-transformer model to convert text chunks into
    numerical vector embeddings. It caches these embeddings to speed up subsequent retrieval calls.

4.  **Semantic Retrieval**: Given a query (e.g., a line from an inspection report), it calculates
    the semantic similarity (cosine similarity) between the query's embedding and all cached
    chunk embeddings. It then returns the text of the top 'k' most relevant chunks.

This retrieved context is then passed to the ExtractorAgent to provide it with relevant
background information, improving the accuracy of its information extraction task.
"""
from typing import List, Tuple
import PyPDF2
from sentence_transformers import util
import torch

# Use a pre-trained sentence-transformer model for creating embeddings
from sentence_transformers import SentenceTransformer

# The name of the model to use for RAG embeddings.
RAG_EMBED_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
# Global variable to hold the loaded model instance (singleton pattern).
rag_embed_model = None


def get_rag_embed_model():
    """
    Loads the sentence-transformer model for RAG embeddings if it hasn't been loaded yet.
    This function implements a singleton pattern to ensure the model is loaded only once.
    """
    global rag_embed_model
    if rag_embed_model is None:
        try:
            rag_embed_model = SentenceTransformer(RAG_EMBED_MODEL_NAME)
            print(f"Local RAG embedding model '{RAG_EMBED_MODEL_NAME}' loaded.")
        except Exception as e:
            print(f"Error loading local RAG model '{RAG_EMBED_MODEL_NAME}': {e}")
            print("RAG embedding will not function correctly without a model.")
    return rag_embed_model


# In-memory caches to avoid redundant processing of the same PDF file.
PDF_CACHE = {}      # Caches the extracted text from a PDF path.
CHUNK_CACHE = {}    # Caches the text chunks and their corresponding embeddings.


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a given PDF file. Caches the result to avoid re-reading.
    """
    if pdf_path in PDF_CACHE:
        return PDF_CACHE[pdf_path]

    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        PDF_CACHE[pdf_path] = text
        print(f"Extracted text from PDF: {pdf_path} ({len(text)} chars)")
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""
    return text


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Splits a long text into smaller chunks of a specified size with some overlap.
    The overlap helps to preserve context between chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def get_pdf_chunks_and_embeddings(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Tuple[str, torch.Tensor]]:
    """
    Processes a PDF file by extracting text, chunking it, and generating embeddings for each chunk.
    Caches the resulting chunks and embeddings.
    """
    cache_key = (pdf_path, chunk_size, chunk_overlap)
    if cache_key in CHUNK_CACHE:
        return CHUNK_CACHE[cache_key]

    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return []

    text_chunks = chunk_text(pdf_text, chunk_size, chunk_overlap)

    model = get_rag_embed_model()
    if not model or not text_chunks:
        return []

    try:
        # Encode all chunks at once for efficiency
        chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True)
        chunk_data = list(zip(text_chunks, chunk_embeddings))
        CHUNK_CACHE[cache_key] = chunk_data
        print(f"Generated {len(chunk_data)} chunks and embeddings for {pdf_path}")
        return chunk_data
    except Exception as e:
        print(f"Error generating embeddings for PDF chunks: {e}")
        return []


def retrieve_relevant_chunks(query: str, pdf_path: str, top_k: int = 3) -> str:
    """
    Retrieves the top 'k' most relevant text chunks from a PDF for a given query.
    """
    model = get_rag_embed_model()
    if not model:
        return "Error: RAG embedding model not loaded."

    chunks_with_embeddings = get_pdf_chunks_and_embeddings(pdf_path)
    if not chunks_with_embeddings:
        return "No chunks available for RAG."

    try:
        # Generate embedding for the input query
        query_embedding = model.encode(query, convert_to_tensor=True)
    except Exception as e:
        print(f"Error generating query embedding for RAG: {e}")
        return "Error generating query embedding."

    # Prepare chunk embeddings for similarity calculation
    all_chunk_embeddings = torch.stack([emb for _, emb in chunks_with_embeddings])

    # Compute cosine similarities between the query and all chunks
    cosine_scores = util.cos_sim(query_embedding, all_chunk_embeddings)[0]

    # Find the indices of the top k highest scores
    top_results_indices = torch.topk(cosine_scores, k=min(top_k, len(chunks_with_embeddings)))

    # Format the relevant chunks into a single context string
    relevant_context = ""
    print(f"\n--- RAG Retrieval for query: '{query[:50]}...' ---")
    for i in range(len(top_results_indices.indices)):
        idx = top_results_indices.indices[i].item()
        score = top_results_indices.values[i].item()
        # Only include chunks that meet a minimum relevance threshold
        if score > 0.3:
            relevant_context += f"\n[Relevant PDF Snippet (Score: {score:.2f})]:\n{chunks_with_embeddings[idx][0]}\n---\n"
            print(f"Found relevant chunk (score {score:.2f}): {chunks_with_embeddings[idx][0][:100]}...")
        else:
            print(f"Skipping chunk (score {score:.2f} below threshold): {chunks_with_embeddings[idx][0][:100]}...")

    return relevant_context if relevant_context else "No highly relevant context found in PDF."
