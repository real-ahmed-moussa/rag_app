# 📌 RAG-App: Retrieval-Augmented Question Answering over PDFs

<p align="center">
  <img src="imgs/rag.png" alt="RAG-App" width="300">
</p>

> A small, production-style Retrieval-Augmented Generation (RAG) system that ingests PDF reports, indexes them with BM25 & vector search, and serves answers via a FastAPI endpoint.



## 📖 Overview
This project implements a **RAG pipeline** that lets you:
- Drop PDF documents into a folder.
- Automatically **extract, chunk, and index** their content.
- Query them using a **FastAPI `/ask` endpoint**, powered by the OpenAI API.
- Retrieve answers grounded in your documents using both **BM25** and **dense vector search**.

It is designed as a **portfolio-grade, minimal RAG service**—clean project layout, clear separation of concerns (ingestion, indexing, retrieval, generation), and simple curl examples for testing.



## 🏢 Business Impact
Most organizations have critical knowledge locked in PDFs: reports, manuals, policies, and contracts. This project shows how to:
1. Turn those PDFs into a **searchable knowledge base**.
2. Provide a **single question-answering API** that can be plugged into dashboards, chatbots, or internal tools.
3. Build a **reproducible RAG pipeline** with explicit steps for ingestion, indexing, and serving.

It demonstrates practical skills in:
1. Designing a **modular RAG system**.
2. Combining **classical IR (BM25)** with **vector search**.
3. Exposing the pipeline as an **API** suitable for real-world integration.



## 🚀 Features
✅ **PDF Ingestion Pipeline** – Extracts text from PDFs and writes clean chunks ready for indexing.  
✅ **Hybrid Retrieval** – BM25 index + vector store (FAISS) for robust retrieval.  
✅ **RAG Answer Generation** – Uses retrieved passages plus the OpenAI API to synthesize grounded answers.  
✅ **FastAPI Service** – `/ask` endpoint for programmatic access.  
✅ **Configurable Data Folder** – Drop new PDFs into `data/pdfs/` and rebuild indexes.  
✅ **Portfolio-Ready Layout** – Clear module boundaries (`ingestion/`, `indexing/`, `retriever/`, `generation/`, `app/`).


| Technology            | Purpose                                       |
|-----------------------|-----------------------------------------------|
| `Python`              | Core language                                |
| `FastAPI` + `Uvicorn` | Web API for question answering               |
| `OpenAI API`          | LLM-based answer generation                  |
| `FAISS`               | Dense vector search for semantic retrieval   |
| `BM25` / Whoosh       | Lexical search over document chunks          |
| `PyPDF` / similar     | PDF text extraction                          |

*(Exact libraries may vary slightly depending on your `requirements.txt`.)*



## 📂 Project Structure
<pre>
📦 RAG-App: Retrieval-Augmented Question Answering over PDFs
 ┣ 📂 app
 │  ┣ 📜 __init__.py
 │  ┗ 📜 main_app.py                                # FastAPI app: defines the /ask endpoint, request models, and routing logic
 ┣ 📂 data
 │  ┗ 📂 pdfs                                       # Raw PDFs to ingest; ignored by Git. Pipeline reads PDFs from here.
 ┣ 📂 generation
 │  ┗ 📜 generate_answer.py                         # Takes retrieved text + question → calls OpenAI → returns grounded answer.
 ┣ 📂 indexing
 │  ┗ 📜 bm25_index.py                              # Builds BM25 (Whoosh) index over the cleaned, chunked text.
 ┣ 📂 ingestion
 │  ┣ 📜 chunk_text.py                              # Splits extracted text into overlapping chunks for retrieval.
 │  ┣ 📜 embed_store.py                             # Creates embeddings and builds FAISS vector index.
 │  ┗ 📜 extract_text.py                            # Extracts clean text from PDFs using PyPDF or similar.
 ┣ 📂 retriever
 │  ┗ 📜 query_retriever.py                         # Hybrid retrieval: BM25 + FAISS → merges top-k results for OpenAI.
 ┣ 📂 vectorstores
 │  ┣ 📂 bm25_index                                 # Auto-generated BM25 index files (never edit manually).
 │  ┗ 📂 faiss_topic                                # FAISS vector index + metadata store. 
 ┣ 📜 pipeline_runner.py                            # Main indexing pipeline: extract → chunk → embed → build indexes.
 ┣ 📜 requirements.txt                              # Python dependencies (FastAPI, FAISS, PyPDF, OpenAI SDK, etc.).
 ┣ 📜 keys.env                                      # Environment file for storing API keys (excluded from Git).
 ┗ 📜 README.md                                     # Project documentation.
</pre>



## 🛠️ Getting Started
1️⃣ **Clone the Repository**
<pre>
git clone https://github.com/ahmedmoussa/rag_app.git
cd rag_app
</pre>

2️⃣ **Create and Activate a Virtual Environment & Install Dependencies**
<pre>
python -m venv .venv
source .venv/bin/activate             # On macOS / Linux
pip install -r requirements.txt
</pre>

3️⃣ **Add PDFs to Data Folder**
Place the documents you want to query inside:
<pre>
data/pdfs/
</pre>

Example:
<pre>
data/pdfs/
 ┣ American_Astronomical_Society_Report.pdf
 ┗ Another_Report.pdf
</pre>



## 🧱 Build the Indexes
Run the pipeline to:
1. Extract and clean text from PDFs
2. Chunk text into passages
3. Build the BM25 and FAISS indexes
<pre>
python pipeline_runner.py
</pre>

This will populate:
- retriever/vectorstores/bm25_index/
- faiss_topic/
with the necessary index files.

Whenever you add or replace PDFs, rerun:
<pre>
python pipeline_runner.py
</pre>
to rebuild the indexes.



## 🌐 Run the API
### 1. Set your OpenAI API Key
On macOS / Linux:
<pre>
export OPENAI_API_KEY="sk-..."
export KMP_DUPLICATE_LIB_OK=TRUE
</pre>

On Windows (PowerShell):
<pre>
$env:OPENAI_API_KEY="sk-..."
$env:KMP_DUPLICATE_LIB_OK="TRUE"
</pre>


### 2. Start the FastAPI App
<pre>
uvicorn app.main_app:app --reload
</pre>
By default, the service runs on `http://127.0.0.1:8000`

You can explore the interactive docs at:
- `http://127.0.0.1:8000/docs` (Swagger UI)
- `http://127.0.0.1:8000/redoc` (ReDoc)



## 🔍 Ask Questions (Examples)
Use `curl` to query the `/ask` endpoint:
<pre>
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "In the American Astronomical Society Report, what are the costs of total assets in year 2022?"}'
</pre>

<pre>
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When was Galileo born and when did he die?"}'
</pre>

The API will:
1. Retrieve relevant passages from the BM25 & FAISS indexes.
2. Send them, along with your question, to the OpenAI model.
3. Return a concise answer grounded in the retrieved context.



## 🧩 Architecture at a Glance
### 1. Ingestion
- Read PDFs from data/pdfs/.
- Extract text and split into overlapping chunks.
### 2. Indexing
- Build a BM25 index for lexical matching.
- Build a FAISS index for dense semantic search.
### 3. Retrieval
- Given a question, retrieve top-k chunks from both indices.
- Optionally merge / re-rank to produce a final context set.
### 4. Generation
- Feed the question + retrieved context to the OpenAI model.
- Return an answer along with optional supporting snippets.



## 🧪 Extending the Project
Some natural extensions:
- Swap in different embedding models (e.g., local sentence transformers).
- Add source citation in the response (page number, document name).
- Expose a chat-style UI (Streamlit, React, etc.) instead of only curl.
- Log Q&A pairs to a database for feedback-driven improvement.
- Add evaluation scripts (e.g. retrieval metrics, hallucination checks).



## 📝 License
- This project is shared for portfolio purposes only and may not be used for commercial purposes without permission.
- This project is licensed under the MIT License.

© 2025 Dr. Ahmed Moussa



## 🤝 Contributing
Pull requests are welcome.  
For major changes, please open an issue first to discuss what you would like to change.



## 📫 Contact
For feedback, bugs, or collaboration ideas:
- **GitHub**: [@real-ahmed-moussa](https://github.com/real-ahmed-moussa)  



## ⭐️ Show Your Support
If you find this project useful, consider giving it a ⭐️ on [GitHub](https://github.com/real-ahmed-moussa/rag_app)!
