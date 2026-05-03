from fastapi import FastAPI
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

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Philippine history expert. Answer using ONLY the context below. If not in context, say: \"I don't have that information in the documents.\"",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {input}"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, question_answer_chain)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = qa_chain.invoke({"input": request.question})
    return {
        "answer": result["answer"],
        "sources": list({doc.metadata.get("source", "unknown") for doc in result.get("context", [])})
    }