import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PARQUET = "data/arxiv_subset.parquet"
OUTPUT_FILE = "embeddings/embeddings.npy"
MODEL_NAME = "allenai/specter2_base"

os.makedirs("embeddings", exist_ok=True)

print("Loading dataset...")
df = pd.read_parquet(INPUT_PARQUET)

texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

print(f"Total texts: {len(texts)}")
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Encoding texts...")
embeddings = model.encode(
    texts[:2000],
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Norm of first embedding: {np.linalg.norm(embeddings[0]):.4f}")

np.save(OUTPUT_FILE, embeddings)
print(f"Saved embeddings to {OUTPUT_FILE}")