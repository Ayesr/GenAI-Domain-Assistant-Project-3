#  GenAI Domain Assistant Project (Chatbot System)

##  Overview
This project demonstrates the development of **AI-powered domain-specific chatbots** using Large Language Models (LLMs). The system enables organizations to access and retrieve internal knowledge efficiently through natural language queries.

It includes both:
- Prompt-engineered chatbots (no RAG)
- RAG-based chatbot system (document-aware Q&A)

The focus is on building **intelligent assistants for enterprise knowledge access and user support**.



##  Key Objectives
- Build role-based AI chatbots using LLMs
- Apply prompt engineering for controlled behavior
- Implement Retrieval-Augmented Generation (RAG)
- Enable contextual question answering from documents
- Compare **direct LLM vs RAG-based responses**
- Simulate real-world enterprise AI assistant behavior



##  Chatbots Implemented

###  HR Assistant Chatbot
A domain-specific assistant designed to answer HR policy-related questions.

**Features:**
- Explains company policies (vacation, sick leave, remote work)
- Provides structured HR guidance
- Answers employee benefit queries
- Maintains professional and formal tone
- Avoids hallucination using strict prompting rules



###  Customer Support Chatbot (TechShop)
An AI support assistant for an electronics retail environment.

**Features:**
- Handles customer queries and complaints
- Explains return, shipping, and warranty policies
- Provides empathetic and helpful responses
- Escalates complex issues when needed
- Simulates real-world customer service behavior



###  RAG-Based Knowledge Assistant
An advanced chatbot that retrieves answers from internal documents using a **Retrieval-Augmented Generation (RAG)** pipeline.

**Pipeline:**
- Document Loading
- Text Chunking
- Retrieval (Keyword-based → upgradeable to semantic search)
- Context Injection into LLM
- Answer Generation via OpenRouter / GPT models

**Capabilities:**
- Answers questions based on internal documents
- Reduces hallucination through grounding
- Provides context-aware responses
- Compares:
  - Without RAG (general knowledge)
  - With RAG (document-based answers)



## 🏗 System Architecture

```text
User Query
   ↓
Streamlit UI
   ↓
Embedding Model (OpenRouter API)
   ↓
ChromaDB Vector Search
   ↓
Top-K Relevant Document Chunks
   ↓
Prompt Construction (Context + Query)
   ↓
LLM (GPT-3.5 via OpenRouter)
   ↓
Final Answer
```


##  Technologies Used
- Python 
- OpenAI API / OpenRouter API
- GPT-3.5-Turbo / GPT-4o-mini
- LangChain (document processing utilities)
- Prompt Engineering
- Retrieval-Augmented Generation (RAG)
- Streamlit  (Frontend UI)
- ChromaDB  (Vector Database)
- OpenRouter API  (LLMs + Embeddings)
- GPT-3.5 / GPT models 



##  Key Features
- Role-based chatbot behavior via system prompts
- Domain-specific assistants (HR, Support, Knowledge Base)
- Document-based question answering (RAG pipeline)
- Comparison of LLM vs RAG outputs
- Modular and extensible architecture
- OpenRouter integration for flexible model usage
- Vector databases (ChromaDB)
- Embedding models (OpenRouter)
- Document retrieval pipelines
- Streamlit-based web applications


##  Project Outcomes

- Gained strong understanding of how prompt engineering controls AI behavior and response quality  
- Developed a complete real-world chatbot system using LLMs and vector databases  
- Implemented a full Retrieval-Augmented Generation (RAG) pipeline from scratch  
- Understood semantic search using embeddings and ChromaDB  
- Compared keyword-based search vs semantic search and observed performance differences  
- Reduced hallucinations by grounding responses in retrieved context  
- Built a functional and interactive AI assistant using Streamlit  
- Integrated OpenRouter embeddings and LLM APIs successfully  
- Learned end-to-end pipeline design for AI-powered knowledge systems  
- Strengthened understanding of enterprise-level AI assistant architecture  

---

##  Future Improvements

-  Implement hybrid search (keyword + semantic) for improved retrieval accuracy  
-  Add reranking models to improve relevance of retrieved documents  
-  Enable multi-document reasoning across multiple knowledge sources  
-  Add user authentication and role-based access control  
-  Deploy the application on cloud platforms (Streamlit Cloud / AWS / Azure)  
-  Integrate voice-based interaction (speech-to-text and text-to-speech)  
-  Add analytics dashboard for query tracking and usage insights  
-  Improve UI with advanced chat features (file upload, citations, highlights)  
-  Optimize embedding pipeline for faster indexing and retrieval  
-  Extend system to support PDF, DOCX, and web-based documents  

## LangChain Components Used

The project uses **LangChain utilities** for document processing and data preparation:

- `DirectoryLoader` → Load documents from folder
- `TextLoader` → Read `.txt` files
- `RecursiveCharacterTextSplitter` → Split documents into chunks for embeddings
- `Document abstraction` → Maintain structured text + metadata

 LangChain is used ONLY for preprocessing and ingestion, not for full agent orchestration.


## Embeddings Model

- `text-embedding-3-small` (OpenRouter)
- Converts text → high-dimensional vector representations
- Enables semantic similarity search in ChromaDB

## RAG Pipeline Implementation
Document Loading (LangChain)
`DirectoryLoader("company_docs/", glob="*.txt")`
`TextLoader()`

## Text Chunking (LangChain)
`RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)`

## Embedding + Storage (ChromaDB)
Documents converted into embeddings using OpenRouter API
Stored in persistent ChromaDB vector store
Each chunk stored with metadata + ID

##  Future Improvements
- Replace keyword search with semantic embeddings (FAISS / ChromaDB)
- Add conversation memory for multi-turn dialogue
- Build web interface using Streamlit / Flask
- Deploy on cloud platforms (AWS / Azure / HuggingFace Spaces)
- Add voice-based interaction (speech-to-text / text-to-speech)
- Create analytics dashboard for query insights

## Semantic Retrieval
`collection.query(
    query_texts=[query],
    n_results=3
)`

## LLM Response Generation

`Prompt format:

Context:
[Retrieved Documents]

Question:
User Query

Answer:`
Model:

`GPT-3.5 (OpenRouter API)`

## Streamlit Chat Application
- Features Implemented
- Chat-style UI (like ChatGPT)
- Session-based memory using st.session_state
- Sidebar analytics (document count, chat count)
- Real-time response generation
- Clean modern interface with CSS styling

Keyword Search ❌
- Matches exact words only
- Cannot understand meaning
- Example: "PTO" ≠ "Vacation"

Semantic Search ✔
- Understands meaning
- PTO → Paid Time Off
- WFH → Remote Work
- Finds conceptually similar content

##  Improvements Made in Project

###  1. Fixed Embedding System
- Replaced OpenAI embedding dependency with OpenRouter-based embeddings  
- Implemented a custom embedding function using `text-embedding-3-small`  
- Resolved embedding dimension mismatch issue (1536 vs 3072)  
- Ensured consistent embedding space across ingestion and query time  



###  2. RAG Pipeline Optimization
- Improved prompt engineering with context-aware instructions  
- Enhanced retrieval quality using ChromaDB vector search  
- Reduced hallucination by grounding responses in retrieved documents  
- Structured context formatting for better LLM comprehension  



###  3. Streamlit UI Enhancements
- Added chat history memory using `st.session_state`  
- Improved sidebar with system analytics (documents & messages count)  
- Integrated loading indicators (spinners) for better UX  
- Enhanced UI with custom CSS styling for a modern interface  



###  4. Code Architecture Improvements
- Modularized core functions (`init_chromadb`, `get_rag_response`)  
- Implemented Streamlit caching for faster initialization  
- Separated ingestion pipeline from inference pipeline  
- Improved readability and maintainability of codebase  



###  5. Retrieval Improvements
- Increased and controlled Top-K retrieval mechanism  
- Improved context formatting using clear separators  
- Enhanced query handling robustness for better stability  
- Better semantic matching between query and documents  

## 🖥️ Streamlit Commands & Hosting Guide


### 🚀 Install Streamlit

```pip install streamlit ```

```pip install -r requirements.txt ```

Run Streamlit App (Localhost)

```streamlit run app.py ```

Open in Browser
```http://localhost:8501/ ```

Run on Public Server (VPS / Cloud VM)
```streamlit run app1.py --server.address 0.0.0.0 --server.port 8501 ```
link to my HR Assistant Chatbot:
```https://genai-domain-assistant-project-3-ayesha.streamlit.app/```
##  Conclusion
This project demonstrates how Large Language Models combined with prompt engineering and retrieval systems can be used to build intelligent domain-specific assistants. It showcases the evolution from basic chatbots to **context-aware AI systems capable of enterprise deployment**.



##  Author
**Ayesha Ameer** ,
**email: shaheenaameer2003@gmail.com**
AI/ML & Electromagnetics Researcher | GenAI Developer
