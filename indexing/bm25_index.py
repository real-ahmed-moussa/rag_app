# 1. Import Libraries
import os
from typing import List, Tuple

from whoosh import index as windex
from whoosh.fields import Schema, ID, NUMERIC, TEXT
from whoosh.qparser import MultifieldParser, OrGroup
from langchain_core.documents import Document


# 2. Global Schema Definition (reuse everywhere)
schema = Schema(
    doc_id=ID(stored=True, unique=True),
    source=ID(stored=True),
    page=NUMERIC(stored=True),
    content=TEXT(stored=True),
)


# 3. Function to Create/Open a Whoosh Index on Disk
def _ensure_index(index_dir: str):
    """
    Create/open a Whoosh index on disk.

    Returns:
        whoosh.index.Index: an index object you can use to write or search.
    """
    os.makedirs(index_dir, exist_ok=True)
    
    if not windex.exists_in(index_dir):
        return windex.create_in(index_dir, schema)
    
    return windex.open_dir(index_dir)


# 4. Function to Build/Update BM25 Index
def build_bm25_index(documents: List[Document], index_dir: str) -> str:
    """
    Build/update a BM25 (BM25F under the hood) index from LangChain Documents.

    Returns:
        str: The path to the index directory.
    """
    idx = _ensure_index(index_dir)
    writer = idx.writer(limitmb=256)
    for d in documents:
        md = d.metadata or {}
        writer.update_document(
                                doc_id=md.get("doc_id", ""),
                                source=md.get("source", ""),
                                page=int(md.get("page", 0)),
                                content=d.page_content or ""
                            )
    writer.commit()
    return index_dir


# 5. Function to Run BM25 Search
def search_bm25(query: str, index_dir: str, top_k: int = 60) -> List[Tuple[Document, float]]:
    """
    Run a BM25 search and return LangChain Documents with their relevance scores.

    Returns:
        List[(Document, score)]: Pairs of LangChain Document and BM25 score.
    """
    idx = windex.open_dir(index_dir)
    qp = MultifieldParser(["content"], schema=idx.schema, group=OrGroup)
    q = qp.parse(query)
    docs = []
    with idx.searcher() as s:
        results = s.search(q, limit=top_k)
        for r in results:
            docs.append((
                Document(
                            page_content=r["content"],
                            metadata={"doc_id": r["doc_id"], "source": r["source"], "page": int(r["page"])}
                        ),
                float(r.score),
            ))
    return docs
