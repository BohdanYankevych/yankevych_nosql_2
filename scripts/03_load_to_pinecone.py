import os
import time
import numpy as np
import pandas as pd
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "arxiv-search"

print("Loading data...")
df = pd.read_parquet("data/arxiv_subset.parquet")

print("Loading embeddings...")
embeddings = np.load("embeddings/embeddings.npy")

print(f"Embeddings shape: {embeddings.shape}")

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

vectors = []

print("Preparing vectors...")

for i, row in df.iloc[:2000].iterrows():
    vectors.append({
        "id": str(row["id"]),
        "values": embeddings[i].tolist(),
        "metadata": {
            "title": row["title"][:500],
            "category": row["category"],
            "year": int(row["year"])
        }
    })

print(f"Prepared vectors: {len(vectors)}")

BATCH_SIZE = 100

print("Uploading to Pinecone...")

for i in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[i:i+BATCH_SIZE]

    index.upsert(vectors=batch)

    print(f"Uploaded {i + len(batch)} / {len(vectors)}")

    time.sleep(0.1)

print("Upload completed!")