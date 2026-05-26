# Arxiv Semantic Search Platform (Vector Databases)

## 📌 Опис проєкту

У цьому проєкті реалізовано систему семантичного пошуку на основі vector database Pinecone та embedding-моделі SPECTER2.

Було виконано:

- Завантаження та підготовка arXiv dataset
- Генерація embedding-векторів
- Завантаження векторів у Pinecone
- Семантичний пошук
- Chunking текстів
- Hybrid Search (BM25 + Vector Search)

---

# ⚙️ Налаштування та запуск

## 1. Створення virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 2. Встановлення залежностей

```bash
pip install -r requirements.txt
```

---

## 3. Налаштування `.env`

Створити файл `.env`

```env
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX=arxiv-search
```

---

# 📂 Структура проєкту

```text
.
├── data/
│   ├── arxiv-metadata-oai-snapshot.json
│   └── arxiv_subset.parquet
│
├── embeddings/
│   └── embeddings.npy
│
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_embed.py
│   ├── 03_load_to_pinecone.py
│   ├── 04_search.py
│   ├── 05_chunking.py
│   └── 06_hybrid_search.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

---

# 🚀 Запуск проєкту

## 1. Підготовка датасету

```bash
python scripts/01_prepare_data.py
```

### Що робить скрипт

- читає arXiv dataset
- бере перші 10000 статей
- очищає дані
- створює parquet dataset

### Результат

```text
data/arxiv_subset.parquet
```

---

## 2. Генерація embedding-векторів

```bash
python scripts/02_embed.py
```

### Що робить

- завантажує модель `allenai/specter2_base`
- створює embeddings для abstracts
- зберігає embeddings

### Результат

```text
embeddings/embeddings.npy
```

### Розмір embeddings

```text
(2000, 768)
```

---

## 3. Завантаження у Pinecone

```bash
python scripts/03_load_to_pinecone.py
```

### Що робить

- створює Pinecone vectors
- додає metadata
- виконує upsert у vector database

---

## 4. Семантичний пошук

```bash
python scripts/04_search.py
```

### Приклад запиту

```text
quantum gravity black holes
```

### Приклад результату

```text
1. BPS Black Holes
2. Tunnelling from black holes and tunnelling into white holes
3. Black Hole's Life at colliders
```

---

# ✂️ Chunking

## Запуск

```bash
python scripts/05_chunking.py
```

---

## Що реалізовано

### Fixed Chunking

Текст ділиться на однакові шматки фіксованого розміру.

### Переваги

- простота
- швидкість

### Недоліки

- може розривати зміст

---

### Semantic Chunking

Текст ділиться логічно за змістом.

### Переваги

- кращий semantic retrieval
- кращий контекст

### Недоліки

- складніша реалізація

---

## Результати chunking

```text
Documents processed: 30
Fixed chunks: 158
Semantic chunks: 143
```

---

# 🔍 Hybrid Search

## Запуск

```bash
python scripts/06_hybrid_search.py
```

---

## Реалізовано

### BM25 Search

Лексичний пошук за ключовими словами.

### Переваги

- добре працює для exact matches

### Недоліки

- не розуміє зміст

---

### Vector Search

Семантичний пошук через embeddings.

### Переваги

- розуміє контекст
- знаходить semantic similarity

### Недоліки

- іноді менш точний для exact keywords

---

### Hybrid Search (RRF)

Поєднання BM25 + Vector Search через Reciprocal Rank Fusion.

### Переваги

- краща якість результатів
- баланс semantic + lexical search

---

# 📊 Приклад Hybrid Search

## Query

```text
black holes quantum gravity
```

### BM25 добре знаходить exact matches

```text
Hawking radiation of linear dilaton black holes
```

### Vector Search знаходить semantic similarity

```text
BPS Black Holes
```

### Hybrid Search комбінує обидва підходи

---

# 🧠 Теоретичні висновки

## Чому vector databases важливі?

Vector databases дозволяють:

- semantic search
- recommendation systems
- RAG systems
- AI retrieval pipelines

---

## Чому embeddings кращі за keyword search?

### Keyword search

- шукає exact words

### Embeddings

- розуміють зміст тексту
- дозволяють semantic similarity search

---

## Для чого потрібен chunking?

LLM та embeddings мають обмеження на розмір контексту.

Chunking:

- ділить текст
- покращує retrieval
- дозволяє працювати з великими документами

---

# ✅ Висновки

У проєкті реалізовано повний pipeline vector search системи:

- data preparation
- embedding generation
- vector database indexing
- semantic retrieval
- chunking
- hybrid search

Було показано, що hybrid search дає кращі результати, ніж окремо BM25 або vector search.

---

# 👨‍💻 Технології

- Python
- Pinecone
- Sentence Transformers
- SPECTER2
- Pandas
- NumPy
- LangChain
- Rank-BM25
- Vector Databases