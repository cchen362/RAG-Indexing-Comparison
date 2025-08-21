# Product Requirement Document (PRD)  
**Project:** RAG Indexing Comparison App  
**Platform:** Streamlit (Phase 1)  

---

## 1. Purpose
The goal is to build a lightweight app to **test and compare indexing strategies in a Retrieval-Augmented Generation (RAG) pipeline**.  
The app allows users to:  
- Upload documents (PDF, DOCX, TXT, etc.).  
- Select which embedding/retrieval/generation components to use via sidebar toggles.  
- Submit a query and see results from all selected methods side-by-side.  
- Log all runs automatically into a Google Sheet for offline analysis.  

---

## 2. Core Features

### 2.1 Sidebar Controls
- **Toggles / multiselect checkboxes** for:  
  - Embedding models (e.g., OpenAI, Cohere, HuggingFace).  
  - Retriever options (Vector-only, BM25, Hybrid).  
  - Reranker (on/off, with model choice).  
  - Generation model (LLM choice).  
- **Chunking strategy parameters** (chunk size, overlap).  
- No API key input fields — keys stored in `.env`.  

### 2.2 Document Upload
- Upload section supporting: **PDF, DOCX, TXT** (extendable to others).  
- Extract text and preprocess consistently.  
- Store documents temporarily for indexing during session.  

### 2.3 Query Execution
- User inputs query → pipeline runs in parallel across enabled configs.  
- Each run includes:  
  - Indexing & retrieval with selected settings.  
  - Reranker step (if toggled).  
  - Response generation with chosen LLM.  

### 2.4 Results Display (on screen)
For each enabled config, show in a **separate card/panel**:  
- **Pipeline components used** (embedding, retriever, reranker, generator).  
- **Timing metrics**:  
  - Total time (query → response).  
  - Breakdown per stage if available.  
- **Response text**.  
- Optional: retrieved top-k chunks/snippets.  

### 2.5 Logging (background, not visible in app)
Every run appends a row to a **Google Sheet** with:  
- Timestamp, Run ID.  
- Query text.  
- Components used (embedding, retriever, reranker, generator, chunking params).  
- Timing (total + breakdown).  
- Response text.  
- Optional: token usage/costs, retrieved doc references.  

---

## 3. Non-Functional Requirements
- **Lightweight & efficient**: Avoid heavy frontend; Streamlit is chosen for speed.  
- **Visually clean**: Use Streamlit theming; minimal CSS overrides if needed.  
- **Robust parsing**: Handle large PDFs & DOCX gracefully (set max file size).  
- **Reproducibility**: Log pipeline config alongside results.  

---

## 4. Tech Stack

### 4.1 Frontend/UI
- **Streamlit** (Python-first, sidebar toggles, file upload, cards/panels).  

### 4.2 Backend/Processing
- Python (directly in Streamlit app).  
- Libraries:  
  - **LangChain/Haystack** (pipeline orchestration).  
  - **FAISS/Weaviate/Pinecone** (vector DB).  
  - **PyPDF2 / python-docx** (file parsing).  
  - **gspread** (Google Sheets logging).  

### 4.3 Config & Secrets
- `.env` file for API keys (OpenAI, Cohere, etc.).  
- Use `python-dotenv` to load keys.  

### 4.4 Logging
- Google Sheets API + `gspread`.  
- Schema:  

| Timestamp | Run ID | Query | Embedding Model | Chunk Params | Retriever | Reranker | Generator | Time (Total) | Time (Stage-wise) | Response | Metadata |  

---

## 5. Architecture Overview

```mermaid
flowchart TD
    A[User] --> B[Streamlit UI]
    B --> C[Sidebar Controls<br/>- Embedding toggle<br/>- Retriever toggle<br/>- Reranker<br/>- Generator]
    B --> D[File Upload<br/>(PDF, DOCX, TXT)]
    B --> E[Query Input]

    C --> F[Pipeline Execution]
    D --> F
    E --> F

    F --> G1[Embedding Model 1]
    F --> G2[Embedding Model 2]
    F --> G3[Retriever Variants]
    F --> G4[LLM Generator]

    G1 --> H[Results Display]
    G2 --> H
    G3 --> H
    G4 --> H

    F --> I[Google Sheets Logging]
    H --> J[User Sees Comparison Results]
```

- Single app handles both frontend and backend logic.  
- Parallel runs triggered for each enabled config.  
- Results displayed and logged seamlessly.  

---

## 6. Example Google Sheet Schema

| Timestamp           | Run ID  | Query              | Embedding Model | Chunk Params | Retriever | Reranker | Generator  | Time (Total) | Time (Stage-wise)          | Response                           | Metadata                   |
|---------------------|--------|--------------------|-----------------|--------------|-----------|----------|------------|--------------|-----------------------------|-----------------------------------|----------------------------|
| 2025-08-21 10:30:00 | run001 | "What is RAG?"     | OpenAI-Ada-002  | size=500,50  | VectorDB  | Off      | GPT-4      | 3.2s         | ingest=0.5s;retr=0.7s;gen=2s | "RAG stands for ..."              | Retrieved chunks: doc1,doc2|
| 2025-08-21 10:32:15 | run002 | "What is RAG?"     | Cohere-Embed    | size=500,50  | BM25      | CrossEnc | GPT-4      | 4.5s         | ingest=0.5s;retr=1.2s;gen=2.8s | "Retrieval Augmented Generation..."| Retrieved chunks: doc3,doc4|

- Each row = one pipeline execution.  
- Run IDs uniquely identify experiment runs.  
- Metadata can include retrieved doc IDs/snippets, cost/token usage, etc.  

---

## 7. MVP Scope (Phase 1)
- Sidebar toggles for embeddings, retrievers, generator LLM.  
- File upload for PDF, DOCX, TXT.  
- Query → run pipeline → show results in cards.  
- Log results into Google Sheets.  
- Basic timing instrumentation.  

---

## 8. Future Enhancements (Phase 2+)
- Add advanced retrievers (hybrid, multi-vector, dense+BM25).  
- Add rerankers with different models.  
- Visualization: latency timelines, cost comparisons.  
- Support larger file formats (CSV, HTML).  
- Export results directly from UI (CSV/Excel).  
- Option for **React + FastAPI** migration for advanced UI.  

---

## 9. Risks & Mitigations
- **API limits** → Cache results, exponential backoff.  
- **Large docs** → Preprocess with chunking & limit size.  
- **Sheet quota** → Rotate sheets monthly to avoid hitting Google Sheets limits.  

---

## 10. Milestones
1. **MVP Build (4–6 weeks)**  
   - Streamlit app with sidebar toggles + file upload + query input.  
   - Run pipeline with 2–3 embedding models, simple retrievers, LLM response.  
   - Logging to Google Sheets.  
2. **Iteration (2–4 weeks)**  
   - Add reranker support, chunk size options, hybrid retrievers.  
   - Improve UI layout (tabs/cards).  
3. **Phase 2 (TBD)**  
   - Scalability improvements, advanced visualizations, possible React migration.  

---
