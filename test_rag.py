from src.genai.rag_store import get_grounding_context

context = get_grounding_context("PNEUMONIA", 0.95)

print(f"\nTOTAL DOCUMENTS RETRIEVED: {len(context)}\n")
print("=" * 60)
for i, doc in enumerate(context, 1):
    print(f"{i}. [{doc['id']}] (score: {doc['relevance_score']})")
print("=" * 60)
