from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a Philippine history expert.
Answer using ONLY the context below.
If not in context, say: "I don't have that information in the documents."

Context:
{context}

Question: {question}
Answer:"""
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = qa_chain.invoke({"query": request.question})
    return {
        "answer": result["result"],
        "sources": list(set([doc.metadata.get("source", "unknown") for doc in result["source_documents"]]))
    }