import os
import sys

# =========================================================
# 🔥 CRITICAL FIXES (MUST BE FIRST)
# Fixes system crashes on Streamlit Cloud hosting
# =========================================================
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Silences background telemetry log alerts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # Fixes protobuf crash

# Fixes outdated Linux SQLite engines on Streamlit Cloud containers
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
import chromadb
from chromadb.config import Settings
import requests
import pypdf
from chromadb.api.types import EmbeddingFunction
from langchain_openai import ChatOpenAI

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Company RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fetch from Streamlit Cloud Secrets Manager securely
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# =========================================================
# EMBEDDING FUNCTION (OPENROUTER SAFE)
# =========================================================
class OpenRouterEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key, model="openai/text-embedding-3-small"):
        self.api_key = api_key
        self.model = model

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]

        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": input
            }
        )

        if response.status_code != 200:
            raise ValueError(f"Embedding Error: {response.text}")

        data = response.json()["data"]
        return [
            [float(x) for x in item["embedding"]]
            for item in data
        ]

# =========================================================
# CHROMADB INIT (Safe Singleton Configuration)
# =========================================================
@st.cache_resource
def init_chromadb():
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    embedding_fn = OpenRouterEmbeddingFunction(
        api_key=OPENROUTER_API_KEY,
        model="openai/text-embedding-3-small"
    )
    return client.get_or_create_collection(
        name="company_docs",
        embedding_function=embedding_fn
    )

# =========================================================
# LLM INIT
# =========================================================
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        model="meta-llama/llama-3-70b-instruct",
        temperature=0.2,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        # Tricks Pydantic into bypassing local token structure validations
        openai_api_key="placeholder-to-bypass-pydantic-validation"
    )

collection = init_chromadb()
llm = init_llm()

# =========================================================
# RAG FUNCTION (SAFE VERSION)
# =========================================================
def get_rag_response(query, n_results=3):
    try:
        # Check if database has files before querying to prevent index panics
        if collection.count() == 0:
            return "⚠️ The knowledge base is currently empty. Please drop policy files into the folder uploader in the sidebar."

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        docs = results.get("documents")

        if not docs or not docs[0] or not docs[0][0].strip():
            return "❌ No relevant information found in our company documents."

        docs = docs[0]
        context = "\n\n---\n\n".join(docs)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional HR assistant. Use ONLY the provided context "
                    "to answer questions. If the answer cannot be found or deduced from the "
                    "context, state clearly that you do not know."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer clearly and professionally:"
            }
        ]

        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"⚠️ Error processing request: {str(e)}"

# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# SIDEBAR (FOLDER & MULTI-FILE DOCUMENT UPLOADER)
# =========================================================
with st.sidebar:
    st.title("🤖 AI HR Assistant")
    st.caption("RAG Pipeline Engine")
    
    st.markdown("""
    - **Search Engine:** Semantic Vector Match
    - **Database:** Local ChromaDB Context
    - **LLM Endpoint:** OpenRouter Gateway
    """)
    st.divider()

    # 🏢 DIRECT FOLDER/FILE INGESTION CONTROL INTERFACE
    # 🏢 DIRECT FOLDER/FILE INGESTION CONTROL INTERFACE
    st.subheader("Upload Company Knowledge")
    
    # Drag and drop or browse multiple files simultaneously
    uploaded_files = st.file_uploader(
        "Select PDF or TXT files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
