# SIKAP (Sistem Kebijakan Analitik Pemerintah) — Interactive Dashboard (Prototype)

This prototype can read legal documents (PDF/Word), perform semantic searches, answer policy questions, classify simple impacts, and generate reports. It's suitable for demos to regional apparatus organizations or leaders.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate | Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Notes
- Supported formats: .pdf / .docx.
- The embedding model will be downloaded on the first run (requires an internet connection).
- The FAISS index and ML model are stored in the `storage/` folder.

## Future Development
- Extraction of legal structures (articles/clauses) + document metadata.
- Cross-validation & hyperparameter tuning for the classification model.
- Recommendation reasoning + impact scoring (qualitative → quantitative).
- Export reports to PDF (currently markdown/HTML).
- Role-based access (Admin vs. Head of Regional Apparatus Organization).

## Workflow Explanation

### Upload & Indexing
1.  **Upload Document**: The user uploads a PDF/DOCX file from the dashboard.
2.  **Save to Folder**: The file is securely saved in the `data/` directory using `pathlib`.
3.  **Ingest & Chunking**: The text is extracted and split into smaller chunks.
4.  **Embedding**: The chunks are converted into vector embeddings (using Hugging Face / Ollama).
5.  **Vector DB (FAISS)**: The embeddings and metadata are stored.

### Query & Answering
1.  **User Query**: The user enters a question on the dashboard.
2.  **Query Embedding**: The question is converted into an embedding using the same model.
3.  **Semantic Search**: FAISS finds the most relevant Top-K contexts.
4.  **QA Agent**:
    *   **Mode 1: Extractive Answer** (without LLM, free).
    *   **Mode 2: Generative Answer** (LLM: OpenAI / Gemini / Ollama).
5.  **Display Answer**: The answer is displayed on the dashboard, with references to the source document.
