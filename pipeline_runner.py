# 1. Import Libraries
import os
from dotenv import load_dotenv
from ingestion.extract_text import extract_text_with_pages, extract_text_from_pdfs
from ingestion.chunk_text import split_parent_child, split_with_metadata
from ingestion.embed_store import embed_and_store
from indexing.bm25_index import build_bm25_index


# 2. Define Constants
DATA_DIR = "data/pdfs"
VECTORSTORE_DIR = "vectorstores/faiss_topic"
BM25_DIR = "vectorstores/bm25_index"


# 3. Define Pipeline Function
def run_pipeline():
    print("[1] Starting RAG pipeline...")
    load_dotenv()

    # (1) Load API Key from .env
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment.")
        return

    # (2) Try Page-aware Extraction First
    print("[2] Extracting text from PDFs (page-aware)...")
    pdf_pages = extract_text_with_pages(DATA_DIR)
    if not pdf_pages:
        print("No PDFs found in data/pdfs/. Please add documents first.")
        return

    # (3) Layout/Semantic Child Chunks with Parent/Page Metadata
    print("[3] Splitting into parent/child chunks...")
    all_child_docs = []
    for filename, pages in pdf_pages.items():
        child_docs = split_parent_child(pages, source=filename,
                                        child_tokens=220, overlap_tokens=40)
        all_child_docs.extend(child_docs)

    if not all_child_docs:
        print("No text chunks generated. Check PDF contents.")
        return

    # (4) Dense Embeddings + FAISS
    print("[4] Creating embeddings and saving FAISS index...")
    embed_and_store(all_child_docs, persist_path=VECTORSTORE_DIR, api_key=api_key)

    # (5) BM25 Lexical Index
    print("[5] Building BM25 (Whoosh) index...")
    os.makedirs(BM25_DIR, exist_ok=True)
    build_bm25_index(all_child_docs, index_dir=BM25_DIR)

    # (6) Completion
    print("[6] Pipeline completed successfully!")
    print(f"FAISS at: {VECTORSTORE_DIR}")
    print(f"BM25  at: {BM25_DIR}")


# 4. Execute Pipeline
if __name__ == "__main__":
    run_pipeline()