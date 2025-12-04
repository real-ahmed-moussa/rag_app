# 1. Import Libraries
import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# 2. Optional Lexical Search
try:
    from indexing.bm25_index import search_bm25
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False


# ----- knobs (could move to yaml later) -----
DENSE_TOP_K = 60
BM25_TOP_K = 60
RRF_K = 60               # larger → flatter fusion
FINAL_TOP_K = 5
USE_MULTI_QUERY = True
N_REWRITES = 3
USE_HYDE = True
# -------------------------------------------


load_dotenv()
VECTORSTORE_DIR = "vectorstores/faiss_topic"
BM25_DIR = "vectorstores/bm25_index"


# 3. Function to Load a Persisted Vectorstore
def load_vectorstore(persist_path=VECTORSTORE_DIR) -> FAISS:
    """
    Load a persisted FAISS vector store using OpenAI embeddings.

    Returns:
        A FAISS vectorstore object ready for similarity_search calls.
    """
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(persist_path, embedding_model, allow_dangerous_deserialization=True)


# 4. Function to Perform Dense Vector Retrieval using FAISS
def _dense_search(vs: FAISS, q: str, k: int) -> List[Tuple[Document, float]]:
    """
    Perform dense (vector) retrieval using FAISS.

    Returns:
        List of (Document, score) where score is similarity (higher = closer).
        If `.similarity_search_with_score` is unavailable, scores default to 0.0.
    """
    try:
        return vs.similarity_search_with_score(q, k=k)
    except Exception:
        # Fallback if with_score not available
        docs = vs.similarity_search(q, k=k)
        return [(d, 0.0) for d in docs]


# 5. Function to Generate Multiple Query Rewrites and perform an [Optional] HyDE.
def _generate_rewrites_openai(query: str, n: int = 3, hyde: bool = True) -> List[str]:
    """
    Generate multiple query rewrites and an [optional] HyDE hypothesis.

    Returns:
        A list like: [original_query, paraphrase_1, ..., paraphrase_n, hyde_text?]
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Ask the Model for n Paraphrases (one per line).
    messages = [
                {"role": "system", "content": "You rewrite search queries into diverse paraphrases."},
                {"role": "user", "content": f"Rewrite the following search query into {n} different short variants, one per line:\n\n{query}"}
            ]
    paraphrase_resp = client.chat.completions.create(
                                                        model="gpt-4o-mini",
                                                        messages=messages,
                                                        temperature=0.2,
                                                        max_tokens=150
                                                    )
    
    # Step 2: Clean and De-duplicate Paraphrases
    rewrites = [ln.strip("- ").strip() for ln in paraphrase_resp.choices[0].message.content.splitlines() if ln.strip()]
    rewrites = [r for r in rewrites if r and r.lower() != query.lower()]
    rewrites = rewrites[:n]

    # Step 3: Generate a Tiny HyDE “mini-answer” - acts as a semantic probe!
    if hyde:
        hyde_resp = client.chat.completions.create(
                                                    model="gpt-4o-mini",
                                                    messages=[
                                                                {"role": "system", "content": "Write a 2-3 sentence factual mini-answer that could answer the question."},
                                                                {"role": "user", "content": query}
                                                            ],
                                                    temperature=0.2,
                                                    max_tokens=120
                                                )
        hyde_text = hyde_resp.choices[0].message.content.strip()
        rewrites.append(hyde_text)

    # Step 4: Prepend the Original Query and Alway Keep in Candidate Set.
    return [query] + rewrites


# 6. Function to Perform Reciprocal Rank Fusion (RRF)
def _rrf_fuse(ranked_lists: List[List[str]], k: int = RRF_K) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion (RRF) across multiple ranked lists.

    Returns:
        Dict mapping doc_id -> fused_score (higher is better).
    """
    scores: Dict[str, float] = {}

    # Step 1: Iterate through Each Ranked List
    for lst in ranked_lists:
        # Step 2: Add RRF Contribution for each doc_id
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    return scores


# 7. Function to Perform Hybrid Retrieval
def retrieve_docs(query: str,
                  persist_path=VECTORSTORE_DIR,
                  bm25_dir=BM25_DIR,
                  k: int = FINAL_TOP_K) -> List[Document]:
    """
    Hybrid retrieval with multi-query + optional HyDE and RRF fusion.

    Pipeline:
      Step 1: Load FAISS (dense) index.
      Step 2: Build a set of query variants (multi-query rewrites + HyDE).
      Step 3: For each query variant, run:
              3a) Dense search (FAISS).
              3b) Lexical search (BM25), if available.
      Step 4: Convert each result list into ordered doc_id lists.
      Step 5: Fuse all lists with RRF into a single score per doc_id.
      Step 6: Map top fused ids back to original Documents and return top-k.

    Args:
        query: The user original query.
        persist_path: Path to FAISS vectorstore on disk.
        bm25_dir: Path to BM25 index directory (if present).
        k: Final number of Documents to return after fusion.

    Returns:
        A list of LangChain `Document` objects, ordered by fused relevance.
    """
    # Step 1: Load Dense Vector Store (FAISS)
    vs = load_vectorstore(persist_path)

    # Step 2: Produce Query Variants (multi-query + HyDE)
    queries = [query]
    if USE_MULTI_QUERY:
        try:
            queries = _generate_rewrites_openai(query, n=N_REWRITES, hyde=USE_HYDE)
        except Exception as e:
            # If OpenAI not available, just use original
            queries = [query]

    # Step 3: Prepare Containers for Fusion
    # - doc_bank stores unique doc_id -> Document to prevent duplicates.
    # - ranked_lists holds ordered doc_id lists for RRF.
    doc_bank: Dict[str, Document] = {}
    ranked_lists: List[List[str]] = []

    # Step 4: Run Semantic and BM25 Retrievals (if available) for Each Query Variant
    for q in queries:
        # Step 4.1: Semantic Search
        dense_hits = _dense_search(vs, q, DENSE_TOP_K)
        dense_ids = []

        # Step 4.2: Accumulate Semantic Results: Collect IDs & Cache Documents
        for d, _ in dense_hits:
            doc_id = (d.metadata or {}).get("doc_id") or f"{(d.metadata or {}).get('source','')}-{hash(d.page_content)}"
            dense_ids.append(doc_id)
            if doc_id not in doc_bank:
                doc_bank[doc_id] = d
        ranked_lists.append(dense_ids)

        # Step 4.3: Lexical Search (BM25)
        if HAS_BM25 and os.path.isdir(bm25_dir):
            try:
                lex_hits = search_bm25(q, bm25_dir, top_k=BM25_TOP_K)
                lex_ids = []
                for d, _ in lex_hits:
                    doc_id = (d.metadata or {}).get("doc_id") or f"{(d.metadata or {}).get('source','')}-{hash(d.page_content)}"
                    lex_ids.append(doc_id)
                    if doc_id not in doc_bank:
                        doc_bank[doc_id] = d
                ranked_lists.append(lex_ids)
            except Exception:
                pass  # if index missing/corrupt, just skip
    
    # Step 5: If Everything Failed >> Do Semantic Search on Original Query Only
    if not ranked_lists:
        dense_hits = _dense_search(vs, query, DENSE_TOP_K)
        return [d for d, _ in dense_hits[:k]]

    # Step 6: Fuse All Ranked Lists into a Single Score per doc_id via RRF
    fused = _rrf_fuse(ranked_lists, k=RRF_K)

    # Step 7: Sort doc_ids by Fused Score (desc)
    top_ids = [doc_id for doc_id, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)][:max(k, 30)]

    # Step 8: Map Fused ids back to Document Objects
    top_docs = [doc_bank[i] for i in top_ids][:k]

    return top_docs
