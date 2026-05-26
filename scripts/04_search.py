import os
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "arxiv-search"

print("Loading embedding model...")
model = SentenceTransformer("allenai/specter2_base")

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

query = input("Enter search query: ")

print("Creating query embedding...")
query_embedding = model.encode(
    query,
    normalize_embeddings=True
).tolist()

print("Searching...")

results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

print("\nTop results:\n")

for i, match in enumerate(results["matches"], start=1):
    metadata = match["metadata"]

    print(f"{i}. Score: {match['score']:.4f}")
    print(f"Title: {metadata['title']}")
    print(f"Category: {metadata['category']}")
    print(f"Year: {metadata['year']}")
    print("-" * 60)
    print("\nFiltered search: category hep-th")

filtered_results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    filter={
        "category": {"$eq": "hep-th"}
    }
)

for i, match in enumerate(filtered_results["matches"], start=1):
    metadata = match["metadata"]

    print(f"{i}. Score: {match['score']:.4f}")
    print(f"Title: {metadata['title']}")
    print(f"Category: {metadata['category']}")
    print(f"Year: {metadata['year']}")
    print("-" * 60)