import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever.query_retriever import retrieve_docs

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

VECTORSTORE_DIR = "vectorstores/faiss_topic"

def build_prompt(query, retrieved_docs):
    """
    Build a prompt combining the query and retrieved context.
    """
    context = "\n\n".join([f"Source: {doc.metadata['source']}\nContent: {doc.page_content}"
                           for doc in retrieved_docs])
    
    prompt = f"""
                    You are a knowledgeable assistant specialized in scientific research.
                    You will answer questions based on the provided context from internal documents.

                    Context: {context}

                    Question: {query}

                    Answer concisely and accurately based ONLY on the context above. If the answer is not in the context, say "I don't have that information from the provided documents."
            """
    return prompt

def generate_answer(query, persist_path=VECTORSTORE_DIR, k=5, model="gpt-3.5-turbo"):
    """
    Retrieves relevant docs and generates an answer using OpenAI's Chat model.
    """
    retrieved_docs = retrieve_docs(query, persist_path, k)

    if not retrieved_docs:
        return "No relevant documents found."

    prompt = build_prompt(query, retrieved_docs)

    response = client.chat.completions.create(
        model=model,
        messages=[
                        {"role": "system", "content": "You are a helpful assistant for scientific domain experts."},
                        {"role": "user", "content": prompt}
                ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    user_query = "What are the proper food storage temperature guidelines?"
    answer = generate_answer(user_query)
    print("\n🤖 Answer:", answer)
