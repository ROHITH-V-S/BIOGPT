# 🧬 BioGPT Explorer

> A production-grade RAG pipeline for biomedical literature search and question answering

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![React](https://img.shields.io/badge/React-18-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview
BioGPT Explorer is an advanced Retrieval-Augmented Generation (RAG) application designed to search and analyze biomedical literature. It leverages modern vector search and biomedical entity recognition to provide accurate, context-aware answers to complex medical questions.

## ✨ Key Features
- **RAG pipeline with FAISS vector search**: Fast and efficient in-memory similarity search for document retrieval.
- **SSE streaming responses**: Real-time generation feedback for a snappy user experience.
- **Entity-aware retrieval with biomedical NER**: Utilizes scispaCy for domain-specific entity extraction to boost retrieval relevance.
- **Multiple embedding backends**: Supports OpenRouter, PubMedBERT, and MiniLM for flexible encoding strategies.
- **Evaluation harness with RAGAS-style metrics**: Built-in tools to measure retrieval and generation performance.
- **Docker Compose deployment**: Containerized for seamless, reproducible setups.
- **Free-tier only**: Architected to run without expensive paid APIs.

## 🏗️ Architecture
```mermaid
flowchart TD
    User([User]) --> Frontend[Frontend (React/Vite)]
    Frontend --> Backend[Backend (FastAPI)]
    Backend --> Engine[RAG Engine]
    Engine --> FAISS[(FAISS Vector Store)]
    Engine --> LLM[LLM (OpenRouter)]
    Engine --> PubMed[PubMed API]
    
    subgraph Data Flow
        FAISS
        LLM
        PubMed
    end
```

## 🚀 Quick Start
### Prerequisites
- Python 3.11+
- Node.js 20+
- OpenRouter API key (free tier)

### Local Development
1. **Clone repo**:
   ```bash
   git clone https://github.com/yourusername/biogpt-explorer.git
   cd biogpt-explorer
   ```
2. **Backend setup**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   # Add your OPENROUTER_API_KEY to .env
   make dev
   ```
3. **Frontend setup**:
   ```bash
   cd ../frontend
   npm install
   npm run dev
   ```
4. **Ingest sample data**:
   Use the `/ingest` API or the UI to load biomedical documents into the vector store.
5. **Open browser**:
   Navigate to `http://localhost:5173` to interact with BioGPT Explorer!

### Docker
```bash
cp .env.example .env
# Edit .env with your OpenRouter key
docker compose up --build
```

## 📊 Evaluation
Our built-in evaluation harness uses RAGAS-style metrics to assess answer quality, retrieval context relevance, and faithfulness. For full details on the evaluation methodology, see [docs/eval_report.md](docs/eval_report.md).

## 🔬 Retrieval Comparison
We've compared different embedding strategies (e.g., PubMedBERT vs MiniLM) to optimize retrieval performance on biomedical data. Read the full analysis at [docs/retrieval_comparison.md](docs/retrieval_comparison.md).

## 🧪 Testing
Run the backend test suite:
```bash
cd backend && make test
```

## 📁 Project Structure
```text
.
├── backend/
│   ├── app/                # FastAPI application
│   │   ├── config.py       # Pydantic settings
│   │   ├── main.py         # App entrypoint
│   │   ├── llm.py          # OpenRouter integration
│   │   ├── embeddings.py   # Embedding models
│   │   ├── ner.py          # scispaCy NER
│   │   ├── rag/            # RAG engine (embedder, loader, engine)
│   ├── requirements.txt    # Python dependencies
│   ├── Makefile            # Build and dev commands
├── frontend/               # Vite/React SPA
│   ├── src/                # UI components and app logic
│   ├── vite.config.ts      # Vite configuration
│   ├── package.json        # Node dependencies
├── docs/                   # Documentation and reports
└── docker-compose.yml      # Docker deployment
```

## 🛠️ Configuration
| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key for LLM calls | (Required) |
| `API_PORT` | Port for the FastAPI backend | `8000` |
| `EMBEDDING_MODEL` | Default embedding model (e.g., `all-MiniLM-L6-v2`) | `all-MiniLM-L6-v2` |
| `FAISS_INDEX_PATH` | Path to save/load the FAISS index | `data/vector.index` |
| `LOG_LEVEL` | Application logging level | `INFO` |

## 📝 API Reference
- `GET /health` - Healthcheck endpoint to verify API and RAG engine status.
- `POST /query` - Main Q&A endpoint. Accepts a user query and returns a synthesized answer (supports SSE streaming).
- `POST /ingest` - Processes a list of PMIDs or raw text to generate embeddings and populate the FAISS index.

## License
MIT
