# 1. Import Libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# 2. Function to Split Text into Chunks and Adds Metadata.
def split_with_metadata(text: str, source: str, chunk_size=500, chunk_overlap=100) -> list[Document]:
    """
    Splits text into chunks and adds metadata (source filename and chunk index).

    Returns a list of LangChain Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                separators=["\n\n", "\n", ".", " ", ""]
                                            )
    chunks = splitter.split_text(text)
    
    documents = [
                    Document(
                                page_content=chunk,
                                metadata={"source": source, "chunk_id": i}
                            )
                    for i, chunk in enumerate(chunks)
    ]
    
    return documents


# 3. Function to Split Text into Chunks and Adds Metadata.
def split_parent_child(pages: list[dict], source: str, child_tokens: int = 220, overlap_tokens: int = 40) -> list[Document]:
    """
    Splits text into chunks and adds metadata (source filename and chunk index).

    Returns child chunks with rich metadata:
    - doc_id: unique id per chunk
    - parent_id: source::p{page}
    - page: page number (1-based)
    - source: original filename
    """
    splitter = RecursiveCharacterTextSplitter(
                                                chunk_size=child_tokens,
                                                chunk_overlap=overlap_tokens,
                                                separators=["\n\n", "\n", ".", " ", ""]
                                            )

    docs: list[Document] = []
    for p in pages:
        page_no = p["page_number"]
        parent_id = f"{source}::p{page_no}"
        chunks = splitter.split_text(p["text"] or "")
        for i, chunk in enumerate(chunks):
            doc_id = f"{parent_id}::c{i}"
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                                "doc_id": doc_id,
                                "parent_id": parent_id,
                                "source": source,
                                "page": page_no
                            }
                        )
                    )
    return docs