from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.genai.knowledge_base import RADIOLOGY_KNOWLEDGE_BASE


class RadiologyRAGStore:
    def __init__(self, documents=None):
        self.documents = documents or RADIOLOGY_KNOWLEDGE_BASE
        self.texts = [doc["text"] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)


    def retrieve(self, query, top_k=3):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.documents[idx]["id"],
                "topic": self.documents[idx]["topic"],
                "text": self.documents[idx]["text"],
                "relevance_score": float(similarities[idx]),
            })
        return results

def get_grounding_context(predicted_label: str, probability: float, top_k: int = 2) -> list:

    store = RadiologyRAGStore()


    if predicted_label == "PNEUMONIA":
        query = "pneumonia findings consolidation opacity recommendation"
        required_ids = {"doc_pneumonia_findings", "doc_recommendation_pneumonia"}
        excluded_ids = {"doc_normal_findings", "doc_recommendation_normal"}
    else:
        query = "normal clear lung fields recommendation"
        required_ids = {"doc_normal_findings", "doc_recommendation_normal"}
        excluded_ids = {"doc_pneumonia_findings", "doc_recommendation_pneumonia"}

    required_ids |= {"doc_ai_disclaimer", "doc_limitations"}

    raw_retrieved = store.retrieve(query, top_k=top_k)
    
    #retrieved = store.retrieve(query, top_k=2)
    retrieved = [r for r in raw_retrieved if r["id"] not in excluded_ids]
    retrieved_ids = {r["id"] for r in retrieved}
    
    # Always force-include disclaimer + limitations for safety, even if
    # they didn't rank in the top-k by similarity alone.
    #forced_ids = {"doc_ai_disclaimer", "doc_limitations"}
    for doc in RADIOLOGY_KNOWLEDGE_BASE:
        #if doc["id"] in forced_ids and doc["id"] not in [r["id"] for r in retrieved]:
        if doc["id"] in required_ids and doc["id"] not in retrieved_ids:
            retrieved.append({
                "id": doc["id"],
                "topic": doc["topic"],
                "text": doc["text"],
                "relevance_score": None,
            })
            retrieved_ids.add(doc["id"])
            
    return retrieved
