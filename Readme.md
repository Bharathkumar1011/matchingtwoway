# 🚀 Two-Way Hybrid Matching Engine (Investor ↔ Startup)

### A Semantic + Rule-Based + LLM Justification Engine for Intelligent Matchmaking

---

## 🧭 Executive Overview

The **Two-Way Hybrid Matching Engine** bridges investors and startups using a combination of:

1. **Rule-Based Scoring** (quantitative filters: industry, location, funding goals)
2. **Semantic Matching** (vector embeddings using `sentence-transformers`)
3. **LLM Justification Layer** (AI-generated explanations powered by `Groq`)

The engine performs **bi-directional matching**:

* **Startups → Investors**: Finds best investors for a given startup.
* **Investors → Startups**: Finds ideal startups for each investor.

It integrates **Elasticsearch (local or cloud)** as a vector database for similarity search, and uses a hybrid logic to compute a normalized final compatibility score.

---

## 🧩 System Architecture Mindmap

```
┌──────────────────────────────────────────┐
│           Matching Engine (CLI)          │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│         Data Loading Layer (CSV)         │
│  - startups.csv                          │
│  - investors.csv                         │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│   Embedding Layer (SentenceTransformer)   │
│  - Model: all-MiniLM-L6-v2               │
│  - Outputs dense vectors (384 dims)      │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│   Elasticsearch (Vector Index Store)     │
│  - investors index                       │
│  - startups index                        │
│  - dense_vector fields                   │
│  - cosine similarity search              │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│ Hybrid Matching Core                     │
│  - Rule-Based Score (0–100)              │
│  - Semantic Score (cosine → 0–100)       │
│  - Weighted Final Score (70:30 blend)    │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│ Justification Layer (Groq LLM)           │
│  - Generates analytical match reasoning  │
│  - Professional structured report        │
└──────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│        CLI + JSON Output Layer           │
│  - Interactive query menu                │
│  - matches_output.json                   │
└──────────────────────────────────────────┘
```

---

## ⚙️ Installation & Setup

### 1. 🧱 Clone the repository

```bash
git clone https://github.com/your-org/twoway-matcher.git
cd twoway-matcher
```

### 2. 🐍 Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # on macOS/Linux
venv\Scripts\activate        # on Windows
```

### 3. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```
numpy
pandas
sentence-transformers
elasticsearch
python-dotenv
groq
faker
tqdm
scikit-learn
torch
```

---

## 🔐 Environment Configuration (`.env`)

### Local Elasticsearch Setup

```
ES_CLOUD_URL=http://localhost:9200
ES_API_KEY=                     # leave empty for local
ES_INVESTOR_INDEX=investors
ES_STARTUP_INDEX=startups
GROQ_ENABLED=false
```

### Elastic Cloud Setup

```
ES_CLOUD_URL=https://your-project-id.region.gcp.elastic.cloud:443
ES_API_KEY=your_elastic_api_key_here
ES_INVESTOR_INDEX=investors_cloud
ES_STARTUP_INDEX=startups_cloud
GROQ_ENABLED=true
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🧠 Matching Logic Explained

### 🧩 1. Rule-Based Scoring

**Deterministic filters** comparing structured attributes:

* Industry overlap → +40
* Stage alignment → +30
* Location match → +10
* Funding within check range → +20
* Team size fit → +10
* Revenue stage fit → +10
* Portfolio & experience scaling → +20 combined

→ Total normalized to **100 points**.

### 🧩 2. Semantic Similarity (Vector Embeddings)

Each startup/investor description is converted to a **dense vector** using the model `all-MiniLM-L6-v2`.

Elasticsearch computes **cosine similarity**:

```python
semantic_score = ((hit["_score"] + 1) / 2) * 100  # normalize -1→1 to 0→100
```

### 🧩 3. Hybrid Scoring Blend

```python
final_score = 0.7 * rule_score + 0.3 * semantic_score
```

Rationale: rule alignment carries more weight than description semantics.

### 🧩 4. LLM Justification (Groq)

When:

```python
final_score >= JUSTIFICATION_SCORE_THRESHOLD and GROQ_ENABLED=True
```

the LLM generates:

* **Overall Match Summary**
* **Key Justifications**
* **Considerations & Warnings**
* **Final Verdict**

It uses the model: `meta-llama/llama-4-scout-17b-16e-instruct`.

---

## 🔁 Two-Way Matching Flow

### Direction 1: Startups → Investors

```
Startup Description ─▶ Embedding ─▶ ES Vector Search (Investors)
   │                                   │
   ├── Rule-Based Filter ──────────────┘
   │
   └── Weighted Scoring + LLM Justification
```

### Direction 2: Investors → Startups

```
Investor Description ─▶ Embedding ─▶ ES Vector Search (Startups)
   │                                   │
   ├── Rule-Based Filter ──────────────┘
   │
   └── Weighted Scoring + LLM Justification
```

Final merged results = **both directions combined** → sorted by `final_score`.

---

## 🧮 Mathematical Model Summary

| Component      | Formula                     | Range | Weight |
| -------------- | --------------------------- | ----- | ------ |
| Rule Score     | Deterministic heuristics    | 0–100 | 70%    |
| Semantic Score | ((cosine + 1) / 2) × 100    | 0–100 | 30%    |
| Final Score    | 0.7 × rule + 0.3 × semantic | 0–100 | —      |

---

## 🧑‍💻 Usage

### Run the engine

```bash
python matcher.py
```

You’ll see:

```
🚀 Starting Two-Way Hybrid Matching Engine...
🔍 Matching Startups → Investors ...
🔍 Matching Investors → Startups ...
```

When complete:

```
💾 All matches saved to matches_output.json
✅ Total matches (both directions): XXXX
```

### Interactive CLI Menu

```
1. Lookup by Investor
2. Lookup by Startup
3. Specific Startup–Investor Match
4. Show Top Matches Again
5. Exit
```

---

## 📊 Output Files

* `matches_output.json` → structured output of all matches.
* Console display of top 50 results with justification indicators.

Example:

```
Startup              Investor                  Rule     Semantic  Final    Justification
----------------------------------------------------------------------------------------
Newman Labs          Atlas Capital Partners     84.23    66.12     78.42   ✅ Available
```

---

## ⚡ Performance & Optimization

| Area                | Problem                          | Solution                               |
| ------------------- | -------------------------------- | -------------------------------------- |
| Model Warmup        | First embedding computation slow | Preload `SentenceTransformer`          |
| Indexing            | Recreates ES index every run     | Add existence check before deletion    |
| LLM Calls           | Groq API latency per match       | Set `GROQ_ENABLED=false` for test runs |
| Repeated Embeddings | Same text recomputed             | Use caching (`functools.lru_cache`)    |
| Dataset Size        | CSVs with 1000+ rows             | Use smaller subsets for debugging      |

---

## 🧰 Troubleshooting

| Symptom                  | Cause                            | Fix                                   |
| ------------------------ | -------------------------------- | ------------------------------------- |
| `❌ Connection failed`    | Wrong ES endpoint or API key     | Check `ES_CLOUD_URL` and `ES_API_KEY` |
| Slow runtime             | Too many LLM calls               | Disable GROQ or lower threshold       |
| Negative semantic scores | Old normalization logic          | Use `((score + 1)/2)*100`             |
| No justifications        | GROQ disabled or below threshold | Enable `GROQ_ENABLED=true`            |

---

## 🧪 Example .env Modes

### Development (Local Elasticsearch)

```
ES_CLOUD_URL=http://localhost:9200
GROQ_ENABLED=false
```

### Production (Elastic Cloud + Groq)

```
ES_CLOUD_URL=https://<your-cluster>.es.<region>.gcp.elastic-cloud.com:443
ES_API_KEY=xxxxxxxxxxxx
GROQ_ENABLED=true
GROQ_API_KEY=xxxxxxxxxxxx
```

---

## 🧠 Conceptual Flow Summary

```
        +-----------------------+
        |  Load Data (CSV)      |
        +-----------------------+
                    ↓
        +-----------------------+
        |  Compute Embeddings   |
        +-----------------------+
                    ↓
        +-----------------------+
        |  Vector Search (ES)   |
        +-----------------------+
                    ↓
        +-----------------------+
        |  Rule-Based Scoring   |
        +-----------------------+
                    ↓
        +-----------------------+
        |  Final Score (70/30)  |
        +-----------------------+
                    ↓
        +-----------------------+
        |  LLM Justification    |
        +-----------------------+
                    ↓
        +-----------------------+
        |  CLI & JSON Outputs   |
        +-----------------------+
```

---

## 📚 References

* [Elasticsearch Dense Vectors Docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html)
* [Sentence Transformers](https://www.sbert.net/)
* [Groq Cloud API](https://console.groq.com)

---

## 🧾 License

MIT License © 2025 — Developed for Kaushalam Initiative / Govt of Andhra Pradesh
