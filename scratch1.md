# Building a RAG System with Google AI Studio (Free)
### Philippine History Document Collection — Complete Beginner's Guide

---

## What this guide covers

You have 16 text documents about Philippine history. This guide teaches you how to turn them into a working RAG (Retrieval-Augmented Generation) system using **Google AI Studio** — which is free, no credit card needed.

By the end, you will have a program that can answer questions like:

> "Who founded the Katipunan and why?"
> "What happened during the Bataan Death March?"
> "What role did Cardinal Sin play in EDSA?"

...and it will pull those answers directly from your documents, not from the AI's general memory.

---

## How RAG works (before you write a single line of code)

Normal AI: You ask a question → the AI answers from whatever it learned during training. It might be wrong or outdated.

RAG: You ask a question → the system searches your documents for the most relevant passages → it hands those passages to the AI → the AI answers using only what it just read.

Think of it like an open-book exam. The AI is the student. Your documents are the book. RAG makes sure the student reads the right pages before answering.

The stages every RAG system goes through:

```
1. LOAD       Read your documents into memory
2. CHUNK      Split them into smaller, searchable pieces
3. EMBED      Convert each piece into numbers that represent its meaning
4. RETRIEVE   When a question comes in, find the pieces closest in meaning to it
5. GENERATE   Feed those pieces to the AI and ask it to answer
```

---

## Your documents

```
ph_history_rag/
├── 01_philippine_history_timeline.txt
├── 02_jose_rizal_biography.txt
├── 03_people_power_revolution.txt
├── 04_katipunan_and_revolution.txt
├── 05_world_war_ii_philippines.txt
├── 06_spanish_colonization.txt
├── 07_american_period_commonwealth.txt
├── 08_marcos_martial_law.txt
├── 09_precolonial_philippines.txt
├── 10_presidents_of_the_philippines.txt
├── 11_philippine_geography.txt
├── 12_culture_language_religion.txt
├── 13_philippine_economy_history.txt
├── 14_moro_peoples_mindanao.txt
├── 15_philippine_american_relations.txt
└── 16_notable_filipinos.txt
```

16 documents. Each one is a plain `.txt` file — the cleanest format for RAG because there is no formatting noise, just text.

---

## Part 1 — Setup

### Get your free Google AI Studio API key

1. Go to [https://aistudio.google.com](https://aistudio.google.com)
2. Sign in with any Google account
3. Click **"Get API key"** in the left sidebar
4. Click **"Create API key"** → copy it somewhere safe

That key is what lets your code talk to Google's Gemini models. Keep it private — do not paste it directly into your code.

### Set up your project folder

Create this structure on your computer:

```
my_rag_project/
├── ph_history_rag/        ← put all 16 .txt files here
├── ingest.py              ← you will create this (Part 2)
├── query.py               ← you will create this (Part 3)
└── .env                   ← your secret API key goes here
```

### Create the `.env` file

Inside `.env`, write exactly this (replace with your actual key):

```
GOOGLE_API_KEY=your-key-here
```

The `.env` file keeps your key out of your code. Never share this file or upload it to GitHub.

### Install the required libraries

Open your terminal inside `my_rag_project/` and run:

```bash
pip install langchain langchain-google-genai langchain-community faiss-cpu python-dotenv
```

Here is what each one does:

| Library | What it is |
|---|---|
| `langchain` | The main framework. Gives you building blocks — loaders, splitters, chains — so you do not have to wire everything from scratch. |
| `langchain-google-genai` | The LangChain connector for Google's Gemini models. This is what lets you use Google AI Studio inside LangChain. |
| `langchain-community` | Extra loaders and tools made by the LangChain community. You need this for `TextLoader` and `FAISS`. |
| `faiss-cpu` | Facebook's vector search library. Stores your document embeddings and searches them fast. The `-cpu` version runs on any computer without a GPU. |
| `python-dotenv` | Reads your `.env` file and makes the key available to your code without hardcoding it. |

---

## Part 2 — Ingest your documents (run this once)

Create `ingest.py` and copy the code below. Every section has an explanation — read it before running.

```python
# ingest.py
# This script runs ONE TIME. It reads your documents, splits them into chunks,
# converts those chunks into vectors (numbers), and saves them to disk.
# After this runs, you never need to run it again unless you add new documents.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------
load_dotenv()
# load_dotenv() reads your .env file and makes GOOGLE_API_KEY available
# to the rest of your code. Without this line, the Google library will
# not know your key and will throw an authentication error.

# -----------------------------------------------------------------------
# STEP 1: LOAD DOCUMENTS
# -----------------------------------------------------------------------
DATA_DIR = "./ph_history_rag"
documents = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(DATA_DIR, filename)

        loader = TextLoader(filepath, encoding="utf-8")
        # TextLoader is a class from langchain_community.document_loaders.
        # Its job: open a text file and return a LangChain Document object.
        # A Document has two parts:
        #   - page_content: the actual text as a string
        #   - metadata: a dictionary for storing extra info (filename, page, etc.)
        # encoding="utf-8" ensures Filipino characters like ñ, é read correctly.

        docs = loader.load()
        # .load() reads the file and returns a list containing one Document.
        # (It's a list because some loaders like PyPDFLoader return one
        # Document per page — so .load() always returns a list for consistency.)

        for doc in docs:
            doc.metadata["source"] = filename
            # We add the filename to each Document's metadata manually.
            # This is important: later, when the RAG retrieves a chunk to
            # answer a question, we can tell the user WHICH file it came
            # from. Without this, answers have no traceable source.

        documents.extend(docs)

print(f"Loaded {len(documents)} documents")

# -----------------------------------------------------------------------
# STEP 2: SPLIT INTO CHUNKS
# -----------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)
# RecursiveCharacterTextSplitter is one of LangChain's most important classes.
#
# The problem it solves: your documents are thousands of words long, but
# the AI works best with focused, bite-sized pieces. If you send the entire
# document for every question, you waste tokens and bury the relevant part
# in noise. You need to break documents into chunks first.
#
# chunk_size=800 — each chunk is at most 800 characters.
#   800 characters is roughly 2-3 paragraphs. Big enough to contain a
#   complete thought, small enough to stay focused on one topic.
#
# chunk_overlap=100 — consecutive chunks share 100 characters of overlap.
#   Why? Imagine a key fact lands right at the boundary between two chunks.
#   Without overlap, it gets cut in half and loses context on both sides.
#   With overlap, both adjacent chunks contain that sentence intact.
#
# The "Recursive" part means it tries to split on natural boundaries first:
#   \n\n (paragraph breaks) → \n (line breaks) → spaces → characters
# It only goes to a finer split if the chunk is still too large.
# This produces natural-feeling chunks rather than cutting mid-sentence.

chunks = splitter.split_documents(documents)
# split_documents() takes your Document list and returns a new, longer list
# of smaller Documents. Each chunk inherits the metadata from its parent
# (so the "source" filename carries through automatically).

print(f"Created {len(chunks)} chunks from {len(documents)} documents")
# Expect something like: "Created ~350 chunks from 16 documents"

# -----------------------------------------------------------------------
# STEP 3: EMBED AND STORE
# -----------------------------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# GoogleGenerativeAIEmbeddings is from langchain_google_genai.
# An embedding model converts text into a vector — a list of numbers that
# represents the MEANING of the text. Similar meanings produce similar vectors.
#
# Example:
#   "Jose Rizal was executed in 1896"  → [0.23, -0.71, 0.44, 0.09, ...]
#   "Rizal was shot at Bagumbayan"     → [0.21, -0.69, 0.47, 0.11, ...]  ← similar!
#   "The price of rice in Manila"      → [-0.55, 0.12, -0.33, 0.88, ...] ← different
#
# "models/embedding-001" is Google's free embedding model.
# It produces 768-dimensional vectors (768 numbers per text chunk).
# IMPORTANT: You must use the SAME model in both ingest.py and query.py.
# Mixing models produces nonsense similarity scores.

vectorstore = FAISS.from_documents(chunks, embeddings)
# FAISS (Facebook AI Similarity Search) is from langchain_community.vectorstores.
# It is a library built by Meta for searching through large collections of
# vectors very quickly — even millions of them.
#
# FAISS.from_documents() does two things:
#   1. Sends every chunk to Google's embedding API to get its vector.
#      (This is the slow part — ~30-60 seconds for 16 documents.)
#   2. Builds a searchable index from all those vectors in memory.
#
# After this line, you have a database that can answer: "which of my 350
# chunks is most similar in meaning to this question?"

vectorstore.save_local("faiss_index")
# Saves the FAISS index to disk as a folder called faiss_index/.
# It creates two files:
#   faiss_index/index.faiss  ← the actual vector data
#   faiss_index/index.pkl    ← the text content and metadata
#
# Without saving, you'd have to re-embed all documents every single run,
# wasting time and API quota. Save once, load instantly every time after.

print("Done! Index saved to faiss_index/")
print("You can now run query.py to ask questions.")
```

Run it:

```bash
python ingest.py
```

Expected output:

```
Loaded 16 documents
Created ~350 chunks from 16 documents
Done! Index saved to faiss_index/
```

This takes 30–60 seconds because it is sending every chunk to Google's API. You only pay this cost once. After this, `query.py` loads the saved index instantly.

---

## Part 3 — Ask questions (run every time)

Create `query.py`:

```python
# query.py
# This script runs every time you want to ask a question.
# It loads the pre-built index, finds the relevant chunks,
# and asks Gemini to answer using only those chunks.

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# -----------------------------------------------------------------------
# LOAD THE SAVED INDEX
# -----------------------------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# We recreate the same embedding object as ingest.py.
# It must be the SAME model — this is how FAISS knows how to convert
# an incoming question into a vector that can be compared to stored vectors.

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
# FAISS.load_local() reads the two files saved by ingest.py and rebuilds
# the in-memory index. This is nearly instant — no API calls needed.
#
# allow_dangerous_deserialization=True is required because FAISS uses
# Python's pickle format to save metadata. LangChain warns about this
# because pickle files CAN run arbitrary code if they came from an untrusted
# source. Since YOU created this file, it is safe. This flag says: "yes,
# I trust the file I made myself."

# -----------------------------------------------------------------------
# BUILD THE RETRIEVER
# -----------------------------------------------------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
# .as_retriever() wraps the FAISS index in a LangChain Retriever object.
# A Retriever has one job: given a text query, return the most relevant Documents.
#
# Internally, it works like this:
#   1. Takes your question: "Who founded the Katipunan?"
#   2. Converts it to a vector using the same embedding model
#   3. Computes cosine similarity between that vector and all stored vectors
#   4. Returns the top-k most similar chunks
#
# search_type="similarity" — use cosine similarity as the distance metric.
#   Cosine similarity measures the angle between two vectors.
#   Angle near 0° → very similar meaning. Angle near 90° → unrelated.
#
# search_kwargs={"k": 5} — return the top 5 most similar chunks.
#   5 is a good starting point. Too few (k=1) and you might miss context.
#   Too many (k=10+) and you flood the prompt with noise.
#   Adjust based on your results.

# -----------------------------------------------------------------------
# WRITE THE PROMPT
# -----------------------------------------------------------------------
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
# PromptTemplate is from langchain.prompts.
# It is a reusable text template with {variable} placeholders that get
# filled in at runtime before being sent to the AI.
#
# input_variables=["context", "question"] — declares which placeholders exist.
#   {context} → replaced with the retrieved document chunks (all 5 of them,
#               joined together as a single string)
#   {question} → replaced with the user's actual question
#
# The instruction "based ONLY on the context passages" is the most
# important part of this entire program. Without it, Gemini draws on its
# own training knowledge to fill gaps — producing answers that sound
# confident but are not grounded in YOUR documents. This is hallucination.
# This one line is what transforms a chatbot into a proper RAG system.

# -----------------------------------------------------------------------
# CONNECT THE LANGUAGE MODEL
# -----------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)
# ChatGoogleGenerativeAI is from langchain_google_genai.
# This is the actual language model — the "generation" part of RAG.
# It receives the filled-in prompt and writes a response.
#
# model="gemini-1.5-flash" — Google's fast, free model on AI Studio.
#   Gemini Flash is lighter than Gemini Pro but handles Q&A very well.
#   It has a large context window (1 million tokens) so it handles
#   long documents easily. It is free within generous daily limits.
#
# temperature=0 — controls how random or creative the output is.
#   0 = the model always picks the most probable next word.
#       Deterministic, consistent, factual. Best for Q&A.
#   1 = more varied and creative. Better for writing or brainstorming.
#   2 = chaotic. Rarely useful.
#   For RAG over factual documents, always use temperature=0.

# -----------------------------------------------------------------------
# BUILD THE CHAIN
# -----------------------------------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
# RetrievalQA is a LangChain "chain" from langchain.chains.
# A chain is a pre-built workflow that connects multiple components
# into one callable object. You give it a question; it handles the rest.
#
# from_chain_type() is the factory method that assembles the chain.
#
# llm=llm
#   Which language model generates the final answer.
#
# chain_type="stuff"
#   How to handle multiple retrieved chunks before sending to the LLM.
#   "stuff" = take all k chunks and stuff them into the prompt at once,
#   separated by newlines. Simple, fast, and works well for small k values.
#   Other options exist:
#     "map_reduce" — summarize each chunk separately, then combine.
#                    Better when you have many chunks and a small context window.
#     "refine"     — answer using chunk 1, then refine the answer with chunk 2, etc.
#                    More thorough but slower and uses more API calls.
#   Start with "stuff". Only switch if your context window overflows.
#
# retriever=retriever
#   The FAISS retriever that finds relevant chunks for each question.
#
# return_source_documents=True
#   Tells the chain to include the raw retrieved chunks in the result dict.
#   This lets you show users WHERE the answer came from and helps you
#   debug cases where the answer is wrong or incomplete.
#
# chain_type_kwargs={"prompt": prompt}
#   Passes your custom PromptTemplate to the chain. Without this,
#   LangChain uses its own default prompt — which does NOT include
#   the "answer only from context" instruction. Always pass your own prompt.

# -----------------------------------------------------------------------
# ASK A QUESTION
# -----------------------------------------------------------------------
print("\nPhilippine History RAG — type your question below.")
print("Type 'quit' to exit.\n")

while True:
    question = input("Your question: ").strip()

    if question.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    if not question:
        continue

    result = qa_chain.invoke({"query": question})
    # .invoke() runs the full pipeline:
    #   1. Embeds your question into a vector
    #   2. Searches FAISS for the 5 most similar chunks
    #   3. Fills {context} and {question} in your PromptTemplate
    #   4. Sends the filled prompt to Gemini
    #   5. Returns a dict: {"result": "...", "source_documents": [...]}

    print("\nAnswer:")
    print(result["result"])
    # result["result"] is the text answer Gemini generated.

    print("\nSources:")
    seen = set()
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            print(f"  - {source}")
            seen.add(source)
    # result["source_documents"] is the list of Document chunks that were
    # retrieved and passed to Gemini. Printing the source file names shows
    # you exactly which documents were used — so you can verify the answer
    # is grounded in real content, not invented.

    print("\n" + "-" * 60 + "\n")
```

Run it:

```bash
python query.py
```

---

## Part 4 — Questions to test your RAG

All of these are answerable from the 16 documents:

```
Who founded the Katipunan and what was its purpose?
What happened during the Bataan Death March?
Why was Jose Rizal executed by the Spanish?
What were the two novels written by Jose Rizal?
What is bayanihan?
Who was Datu Lapulapu and why is he remembered?
When did the Philippines gain independence and from whom?
What was the People Power Revolution and how did it succeed?
What is the Bangsamoro and how was it created?
Who were the Thomasites?
What is the difference between the MNLF and MILF?
What happened at the Tejeros Convention?
Who was Apolinario Mabini?
What languages are spoken in the Philippines?
What caused Ferdinand Marcos to declare Martial Law?
What is the Laguna Copperplate Inscription?
What was the Manila Galleon Trade?
Who was Gabriela Silang?
How did Ninoy Aquino's assassination lead to the fall of Marcos?
What is the OFW economy?
```

---

## Part 5 — Troubleshooting

**The answer is wrong or the AI is making things up**

This usually means the relevant chunk was not retrieved. Try increasing `k` from 5 to 8 in your retriever. Also verify the information actually exists in your documents — if it is not there, Gemini will sometimes invent an answer anyway despite the prompt instruction.

**"I don't have enough information" but the answer IS in the documents**

The retriever found the wrong chunks. The phrasing of your question might not match the phrasing in the documents. Try rephrasing using words likely to appear in the source text. For example, instead of "How did Marcos lose power?", try "What ended the Marcos dictatorship?" — closer to how the documents are written.

**The answer is too vague or surface-level**

Try decreasing `chunk_size` from 800 to 500 in `ingest.py`. Smaller chunks are more specific. If you change chunk settings, delete the `faiss_index/` folder and re-run `ingest.py` from scratch — old vectors must be regenerated.

**API errors or rate limits**

Google AI Studio's free tier has per-minute rate limits. If you hit them during ingest, add `import time` and `time.sleep(1)` between batches. For querying, just wait a minute and try again.

---

## Part 6 — Adding more documents

1. Drop any `.txt` file into `ph_history_rag/`
2. Delete the `faiss_index/` folder
3. Run `python ingest.py` again

Your `query.py` does not change. The new content is automatically available once the index rebuilds.

Good document habits for RAG:
- **One topic per file** — keeps chunks focused and retrievable
- **Clear section headings** — helps the splitter find natural break points
- **Specific names, dates, numbers** — these are what the retriever matches against
- **Plain prose over tables** — tables often split badly at chunk boundaries

---

## Quick reference

```bash
# Install once
pip install langchain langchain-google-genai langchain-community faiss-cpu python-dotenv

# Run once (or when you add new documents)
python ingest.py

# Run every time you want to ask questions
python query.py
```

| File | Purpose | When to run |
|------|---------|-------------|
| `.env` | Stores your Google API key | Never run — just edit |
| `ingest.py` | Loads, chunks, embeds, saves index | Once, or when docs change |
| `query.py` | Loads index, retrieves, generates answers | Every session |
| `faiss_index/` | The saved vector database | Auto-created by ingest.py |

---

## Library and class summary

| Name | From | What it does |
|------|------|-------------|
| `load_dotenv()` | `python-dotenv` | Reads `.env` file and loads variables into the environment |
| `TextLoader` | `langchain_community` | Opens a `.txt` file and returns a LangChain Document object |
| `RecursiveCharacterTextSplitter` | `langchain` | Splits long Documents into overlapping chunks |
| `GoogleGenerativeAIEmbeddings` | `langchain-google-genai` | Converts text to vectors using Google's embedding model |
| `FAISS` | `langchain_community` | Stores and searches vectors by cosine similarity |
| `ChatGoogleGenerativeAI` | `langchain-google-genai` | Sends prompts to Gemini and returns generated text |
| `PromptTemplate` | `langchain` | A reusable template with `{variable}` placeholders |
| `RetrievalQA` | `langchain` | Pre-built chain: retrieve chunks → fill prompt → generate answer |