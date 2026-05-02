import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

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
print(f"Loaded {len(documents)} documents")






splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks from {len(documents)} documents")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embeddings)


vectorstore.save_local("faiss_index")