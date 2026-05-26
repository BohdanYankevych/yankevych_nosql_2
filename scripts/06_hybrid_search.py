import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-search"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)

df = pd.read_parquet("data/arxiv_subset.parquet").iloc[:2000].reset_index(drop=True)

def tokenize(text):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

corpus = (df["title"] + " " + df["abstract"]).tolist()
tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def bm25_search(query, top_k=TOP_K):
    scores = bm25.get_scores(tokenize(query))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank": rank,
            "id": str(df.iloc[idx]["id"]),
            "title": df.iloc[idx]["title"],
            "score": float(scores[idx]),
            "method": "BM25"
        })
    return results

def vector_search(query, top_k=TOP_K):
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    output = []
    for rank, match in enumerate(results["matches"], start=1):
        output.append({
            "rank": rank,
            "id": match["id"],
            "title": match["metadata"]["title"],
            "score": float(match["score"]),
            "method": "Vector"
        })
    return output

def rrf_fusion(result_lists, k=60, top_k=TOP_K):
    scores = {}

    for results in result_lists:
        for item in results:
            doc_id = item["id"]

            if doc_id not in scores:
                scores[doc_id] = {
                    "id": doc_id,
                    "title": item["title"],
                    "rrf_score": 0
                }

            scores[doc_id]["rrf_score"] += 1 / (k + item["rank"])

    fused = sorted(
        scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return fused[:top_k]

queries = [
    "black holes quantum gravity",
    "quantum chromodynamics photon production",
    "string theory cosmology"
]

for query in queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    bm25_results = bm25_search(query)
    vector_results = vector_search(query)
    hybrid_results = rrf_fusion([bm25_results, vector_results])

    print("\n--- BM25 TOP 5 ---")
    for r in bm25_results:
        print(f"{r['rank']}. {r['title']} | score={r['score']:.4f}")

    print("\n--- VECTOR TOP 5 ---")
    for r in vector_results:
        print(f"{r['rank']}. {r['title']} | score={r['score']:.4f}")

    print("\n--- HYBRID RRF TOP 5 ---")
    for i, r in enumerate(hybrid_results, start=1):
        print(f"{i}. {r['title']} | rrf_score={r['rrf_score']:.4f}")