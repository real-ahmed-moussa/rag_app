# 1. Import Libraries
import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from generation.generate_answer import generate_answer


# 2. Initialize FastAPI App
app = FastAPI(title="Generic RAG API", version="0.0.1")


# 3. Define Request Model
class QueryRequest(BaseModel):
    question: str


# 4. Define API Endpoints
@app.get("/")
def root():
    return {"message": "Welcome to the Generic RAG API!"}


# 5. Endpoint to Handle Questions
@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    Endpoint to get an AI-generated answers from the Generic RAG system.
    """
    answer = generate_answer(request.question)
    return {"question": request.question, "answer": answer}
