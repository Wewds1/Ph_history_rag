from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

app = FastAPI()

# Load embeddings and index once at startup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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