import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})



prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        You are a knowledgeable and helpful assistant specializing in Philippine history.
        Your answers must be based ONLY on the context passages provided below.
        Do not use any outside knowledge.
        If the answer cannot be found in the context, say exactly:
        "I don't have enough information in the provided documents to answer that."

    Context passages from the documents:
        {context}

        Question: {question}

    Answer (based only on the context above):"""
)

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

print("\nPhilippine History RAG — type your question below.")
print("Type 'quit' to exit.\n")



while True:
    question = input("Your question: ").strip()
    if question.lower() == "quit":
        print("Goodbye!")
        break
    
    if not question:
        continue
    
    result = qa_chain.invoke({"query":question})
    print("\nAnswer:")
    print(result["result"])

    print("\nSources:")
    seen = set()
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            print(f"  - {source}")
            seen.add(source)