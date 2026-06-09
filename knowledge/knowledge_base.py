# knowledge_refactored/knowledge_base.py
import json
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict, Optional, Any

DEFAULT_KB_EMBED_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
kb_embed_model = None


def get_kb_embed_model():
    global kb_embed_model
    if kb_embed_model is None:
        try:
            kb_embed_model = SentenceTransformer(DEFAULT_KB_EMBED_MODEL_NAME)
            print(f"Local KB embedding model '{DEFAULT_KB_EMBED_MODEL_NAME}' loaded.")
        except Exception as e:
            print(f"Error loading local KB model '{DEFAULT_KB_EMBED_MODEL_NAME}': {e}")
            print("KnowledgeBase similarity search might not function correctly.")
    return kb_embed_model


def load_json_from_path(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_to_path(data: Any, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _get_embedding_for_kb(text: str) -> Optional[torch.Tensor]:
    model = get_kb_embed_model()
    if model:
        return model.encode(text, convert_to_tensor=True)
    print("Warning: KB embed model not available for_get_embedding_for_kb")
    return None


def calculate_similarity(embedding1: Optional[torch.Tensor],
                         embedding2: Optional[torch.Tensor]) -> float:
    if embedding1 is None or embedding2 is None or embedding1.nelement() == 0 or embedding2.nelement() == 0:
        return 0.0
    if embedding1.shape != embedding2.shape and embedding1.ndim > 0 and embedding2.ndim > 0:
        print(
            f"Warning: Mismatched embedding shapes for similarity calculation: {embedding1.shape} vs {embedding2.shape}")
        if embedding1.ndim == 0 or embedding2.ndim == 0:
            return 0.0
    cosine_sim = util.cos_sim(embedding1, embedding2)
    return cosine_sim.item()


# --- MODIFICATION START ---

def _find_max_similarity(text_embedding: Optional[torch.Tensor],
                         embeddings_list: List[Optional[torch.Tensor]]) -> float:
    """Helper function to find the maximum similarity between a text and a list of existing texts."""
    if text_embedding is None or not embeddings_list:
        return 0.0

    max_similarity = 0.0
    for existing_embedding in embeddings_list:
        if existing_embedding is None:
            continue
        similarity = calculate_similarity(text_embedding, existing_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
    return max_similarity


def is_similar_for_kb(text_embedding: Optional[torch.Tensor], embeddings_list: List[Optional[torch.Tensor]],
                      threshold=0.85) -> bool:
    """
    Checks if a text is similar to any text in a list based on a threshold.
    This version explicitly finds the max similarity first.
    """
    max_similarity = _find_max_similarity(text_embedding, embeddings_list)
    return max_similarity >= threshold


def deduplicate_data_for_kb(data_json: List[Dict[str, Any]], threshold=0.85) -> List[Dict[str, Any]]:
    unique_data = []
    embeddings_list: List[Optional[torch.Tensor]] = []

    for entry in data_json:
        text = entry.get("文本", "")
        if not text:
            unique_data.append(entry)
            continue

        text_embedding = _get_embedding_for_kb(text)

        # Using the rewritten is_similar_for_kb function
        if text_embedding is None or not is_similar_for_kb(text_embedding, embeddings_list, threshold):
            unique_data.append(entry)
            if text_embedding is not None:
                embeddings_list.append(text_embedding)
    return unique_data


def update_knowledge_base_file(data_json: List[Dict[str, Any]], base_json_path: str, similarity_threshold=0.85):
    """
    Updates the knowledge base by adding new, non-similar entries.
    The logic is now based on the novelty score formula you provided.
    Novelty Score = 1 - max_similarity.
    We add an item if its Novelty Score is high enough, which is the same as its max_similarity being low enough.
    """
    try:
        base_json_content = load_json_from_path(base_json_path)
        if not isinstance(base_json_content, list):
            print(f"Warning: Knowledge base file '{base_json_path}' did not contain a list. Initializing as empty.")
            base_json_content = []
    except (FileNotFoundError, json.JSONDecodeError):
        base_json_content = []

    base_texts = [entry.get("文本", "") for entry in base_json_content]
    valid_base_texts = [text for text in base_texts if text]
    valid_base_embeddings: List[Optional[torch.Tensor]] = [_get_embedding_for_kb(text) for text in valid_base_texts]
    valid_base_embeddings = [emb for emb in valid_base_embeddings if emb is not None]

    new_entries_added = 0
    for entry in data_json:
        text = entry.get("文本", "")
        if not text:
            continue

        text_embedding = _get_embedding_for_kb(text)
        if text_embedding is None:
            continue

        # Find the maximum similarity to any existing entry in the knowledge base.
        max_similarity = _find_max_similarity(text_embedding, valid_base_embeddings)

        if max_similarity < similarity_threshold:
            base_json_content.append(entry)
            valid_base_embeddings.append(text_embedding)  # Add its embedding for subsequent comparisons in this run
            new_entries_added += 1

    if new_entries_added > 0:
        save_json_to_path(base_json_content, base_json_path)
        print(f"Knowledge base '{base_json_path}' updated with {new_entries_added} new entries.")
    else:
        print(f"No new unique entries to add to knowledge base '{base_json_path}'.")


# --- MODIFICATION END ---


class KnowledgeBase:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.knowledge: List[Dict[str, Any]] = []
        self.knowledge_embeddings: List[Optional[torch.Tensor]] = []
        self.load_knowledge()

    def load_knowledge(self):
        try:
            loaded_knowledge = load_json_from_path(self.file_path)
            if not isinstance(loaded_knowledge, list):
                print(f"Warning: Knowledge from '{self.file_path}' is not a list. KB will be empty.")
                self.knowledge = []
                self.knowledge_embeddings = []
                return

            self.knowledge = loaded_knowledge
            texts = [example.get('文本', "") for example in self.knowledge]
            valid_texts_with_indices = [(i, text) for i, text in enumerate(texts) if text]
            self.knowledge_embeddings = [None] * len(self.knowledge)

            if valid_texts_with_indices:
                embeddings_for_valid_texts = [_get_embedding_for_kb(text) for _, text in valid_texts_with_indices]
                for (original_idx, _), emb in zip(valid_texts_with_indices, embeddings_for_valid_texts):
                    self.knowledge_embeddings[original_idx] = emb

            print(f"Knowledge base loaded from '{self.file_path}' with {len(self.knowledge)} examples.")
        except (FileNotFoundError, json.JSONDecodeError):
            self.knowledge = []
            self.knowledge_embeddings = []
            print(f"Knowledge base file '{self.file_path}' not found or invalid. Initialized empty KB.")

    def search_similar(self, query_text: str, threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        if not self.knowledge or not get_kb_embed_model() or not query_text:
            return None

        query_embedding = _get_embedding_for_kb(query_text)
        if query_embedding is None:
            return None

        best_match_idx = -1
        highest_similarity = -1.0

        for i, example_embedding in enumerate(self.knowledge_embeddings):
            if example_embedding is None:
                continue
            similarity = calculate_similarity(query_embedding, example_embedding)
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_idx = i

        if best_match_idx != -1 and highest_similarity >= threshold:
            print(f"Found similar example in KB (score: {highest_similarity:.2f}) for query: '{query_text[:50]}...'")
            return self.knowledge[best_match_idx]
        elif best_match_idx != -1:
            print(
                f"Closest match in KB had similarity {highest_similarity:.2f} (below threshold {threshold}) for query: '{query_text[:50]}...'")
        return None
