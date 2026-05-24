import os

# =========================================================
# 🔥 CRITICAL FIX (MUST BE FIRST)
# Fixes protobuf / chromadb crash on Streamlit Cloud
# =========================================================
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import chromadb
import requests
from dotenv import load_dotenv
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

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

# =========================================================
# EMBEDDING FUNCTION (OPENROUTER SAFE)
# =========================================================
class OpenRouterEmbeddingFunction(EmbeddingFunction):

    def __init__(self, api_key, model="openai/text-embedding-3-small"):
        self.api_key = api_key
        self.model = model

    def __call__(self, input):

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

        # =========================================================
        # SAFE FIX: ensure float embeddings (prevents silent crashes)
        # =========================================================
        return [
            [float(x) for x in item["embedding"]]
            for item in data
        ]

# =========================================================
# CHROMADB INIT
# =========================================================
@st.cache_resource
def init_chromadb():

    client = chromadb.PersistentClient(path="./chroma_db")

    embedding_fn = OpenRouterEmbeddingFunction(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="openai/text-embedding-3-small"
    )

    collection = client.get_or_create_collection(
        name="company_docs",
        embedding_function=embedding_fn
    )

    return collection

# =========================================================
# LLM INIT
# =========================================================
@st.cache_resource
def init_llm():

    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

collection = init_chromadb()
llm = init_llm()

# =========================================================
# RAG FUNCTION (SAFE VERSION)
# =========================================================
def get_rag_response(query, n_results=3):

    try:

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        docs = results.get("documents")

        # =========================================================
        # SAFE CHECK (prevents index crash)
        # =========================================================
        if not docs or not docs[0]:
            return "❌ No relevant information found."

        docs = docs[0]

        context = "\n\n---\n\n".join(docs)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional HR assistant. "
                    "Use ONLY the provided context."
                )
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}

Answer clearly and professionally:
"""
            }
        ]

        response = llm.invoke(messages)

        return response.content

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# =========================================================
# SESSION STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("🤖 AI HR Assistant")

    st.markdown("""
- Semantic Search  
- ChromaDB  
- OpenRouter LLM  
- RAG Pipeline  
""")

    st.divider()

    st.metric("📄 Indexed Docs", collection.count())
    st.metric("💬 Messages", len(st.session_state.messages))

    st.divider()

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================================================
# HEADER
# =========================================================
st.title("Company Knowledge Assistant")
st.caption("RAG + Semantic Search + LLM")

# =========================================================
# CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================================================
# CHAT INPUT
# =========================================================
if prompt := st.chat_input("Ask a question about company policies..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            response = get_rag_response(prompt)

        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})