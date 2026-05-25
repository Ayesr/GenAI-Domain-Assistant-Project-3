import os
import sys

# =========================================================
# 1. SYSTEM PATCHES (MUST RUN BEFORE ANY OTHER IMPORTS)
# =========================================================
# Fix the Protobuf descriptor version conflict
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Fix the older SQLite version on Streamlit Cloud hosts
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
import chromadb
import requests
from chromadb.api.types import EmbeddingFunction
from langchain_openai import ChatOpenAI

# Securely grab the API key from Streamlit Secrets
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]


# =========================================================
# 2. LLM INITIALIZATION (Configured safely for OpenRouter)
# =========================================================
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        # This string tricks Pydantic into passing internal OpenAI token validation
        openai_api_key="placeholder-to-bypass-pydantic-validation", 
        model="meta-llama/llama-3-70b-instruct",  
        temperature=0.7
    )


# =========================================================
# 3. EMBEDDINGS (OpenRouter)
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
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": input},
        )

        if response.status_code != 200:
            raise ValueError(f"OpenRouter Embeddings Error: {response.text}")

        data = response.json()["data"]
        return [[float(x) for x in item["embedding"]] for item in data]


# =========================================================
# 4. CHROMADB INITIALIZATION
# =========================================================
@st.cache_resource
def init_chromadb():
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = OpenRouterEmbeddingFunction(api_key=OPENROUTER_API_KEY)

    return client.get_or_create_collection(
        name="company_docs",
        embedding_function=embedding_fn
    )


# Initialize Core Services
collection = init_chromadb()
llm = init_llm()


# =========================================================
# 5. RAG LOGIC
# =========================================================
def get_rag_response(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [[]])[0]

    if not docs:
        return "❌ No relevant information found."

    context = "\n\n---\n\n".join(docs)

    messages = [
        {
            "role": "system",
            "content": "You are a professional HR assistant. Use ONLY the provided context to answer questions."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        }
    ]

    return llm.invoke(messages).content


# =========================================================
# 6. STREAMLIT UI
# =========================================================
st.title("Company Knowledge Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_rag_response(prompt)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
