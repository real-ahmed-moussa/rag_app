# 1. Import Libraries
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# 2. Function to Embed and Store Documents in a VectorDB - Currently FAISS
def embed_and_store(documents, persist_path="faiss_index", api_key=None):
    """
    Embeds the provided documents and stores them in a FAISS index.

    Returns Vectorstore
    """
    if not api_key:
        raise ValueError("OpenAI API key must be provided to embed_and_store")

    embedding_model = OpenAIEmbeddings(
                                        model="text-embedding-3-small",
                                        openai_api_key=api_key
                                    )
    
    vectorstore = FAISS.from_documents(documents, embedding_model)

    os.makedirs(persist_path, exist_ok=True)
    vectorstore.save_local(persist_path)
    print(f"FAISS index saved at: {persist_path}")

    return vectorstore