import re
import pandas as pd

INPUT_FILE = "data/arxiv_subset.parquet"
NUM_DOCS = 30

df = pd.read_parquet(INPUT_FILE)

# беремо 30 найдовших abstract
df = df.copy()
df.loc[:, "abstract_len"] = df["abstract"].str.len()
long_docs = df.sort_values("abstract_len", ascending=False).head(NUM_DOCS)

def fixed_size_chunk(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks

def semantic_chunk(text, max_words=80):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []

    for sentence in sentences:
        sentence_words = sentence.split()

        if len(current) + len(sentence_words) <= max_words:
            current.extend(sentence_words)
        else:
            if current:
                chunks.append(" ".join(current))
            current = sentence_words

    if current:
        chunks.append(" ".join(current))

    return chunks

fixed_chunks = []
semantic_chunks = []

for _, row in long_docs.iterrows():
    fixed = fixed_size_chunk(row["abstract"])
    semantic = semantic_chunk(row["abstract"])

    for i, chunk in enumerate(fixed):
        fixed_chunks.append({
            "arxiv_id": row["id"],
            "title": row["title"],
            "chunk_id": i,
            "chunk": chunk,
            "strategy": "fixed"
        })

    for i, chunk in enumerate(semantic):
        semantic_chunks.append({
            "arxiv_id": row["id"],
            "title": row["title"],
            "chunk_id": i,
            "chunk": chunk,
            "strategy": "semantic"
        })

print("=== Chunking summary ===")
print(f"Documents processed: {NUM_DOCS}")
print(f"Fixed chunks: {len(fixed_chunks)}")
print(f"Semantic chunks: {len(semantic_chunks)}")

print("\n=== Fixed chunk example ===")
print(fixed_chunks[0])

print("\n=== Semantic chunk example ===")
print(semantic_chunks[0])