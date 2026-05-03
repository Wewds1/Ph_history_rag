from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

load_dotenv()

app = FastAPI()

DATA_DIR = "./ph_history"

documents = []
try:
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 5
except Exception as e:
    import logging
    logging.exception("Failed to load documents for retrieval; falling back to empty retriever: %s", e)

    class EmptyRetriever:
        def __init__(self, k=5):
            self.k = k

        def get_relevant_documents(self, query: str):
            return []

    retriever = EmptyRetriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Philippine history expert. Answer using ONLY the context below. If not in context, say: \"I don't have that information in the documents.\"",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {input}"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, question_answer_chain)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        result = qa_chain.invoke({"input": request.question})

        # Normalize common result shapes from different LangChain versions
        if isinstance(result, dict):
            # Try several common keys for the answer text
            answer = (
                result.get("answer")
                or result.get("result")
                or result.get("output_text")
                or result.get("output")
                or None
            )

            # Collect sources from various possible keys
            sources = []
            if "source_documents" in result and isinstance(result["source_documents"], list):
                sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
            elif "context" in result and isinstance(result["context"], list):
                sources = [doc.metadata.get("source", "unknown") for doc in result["context"]]
            elif "sources" in result and isinstance(result["sources"], list):
                sources = result["sources"]

            return {"answer": answer or "", "sources": list(dict.fromkeys(sources))}

        # If result is a plain string or other type, return it as the answer
        return {"answer": str(result), "sources": []}

    except Exception as e:
        import logging, traceback
        logging.exception("Error handling /ask request")
        # Return a concise error message for debugging (do not expose secrets)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")