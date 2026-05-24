import os
import streamlit as st
import chromadb
import requests
from dotenv import load_dotenv
from chromadb.api.types import EmbeddingFunction
from langchain_openai import ChatOpenAI


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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
# CUSTOM CSS (Beautiful UI)
# =========================================================
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background: linear-gradient(to bottom right, #0f172a, #111827);
    color: white;
}

/* Title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.subtitle {
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #374151;
}

/* Sidebar Text */
[data-testid="stSidebar"] * {
    color: white;
}

/* Chat Messages */
.stChatMessage {
    border-radius: 18px;
    padding: 15px;
    margin-bottom: 15px;
    border: 1px solid #374151;
    background-color: rgba(255,255,255,0.03);
    backdrop-filter: blur(6px);
}

/* User Message */
[data-testid="stChatMessageContent"] {
    font-size: 16px;
}

/* Input Box */
.stChatInput input {
    background-color: #1f2937 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #4b5563 !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(to right, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 18px;
    font-weight: 600;
}

.stButton button:hover {
    background: linear-gradient(to right, #1d4ed8, #6d28d9);
    transform: scale(1.02);
    transition: 0.2s ease-in-out;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.04);
    border: 1px solid #374151;
    padding: 15px;
    border-radius: 14px;
}

/* Divider */
hr {
    border-color: #374151;
}

/* Welcome Box */
.welcome-box {
    background: rgba(255,255,255,0.05);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #374151;
    margin-bottom: 20px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #4b5563;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD ENVIRONMENT
# =========================================================
load_dotenv()

# =========================================================
# OPENROUTER EMBEDDING FUNCTION
# =========================================================
class OpenRouterEmbeddingFunction(EmbeddingFunction):

    def __init__(
        self,
        api_key,
        model="openai/text-embedding-3-small"
    ):
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
            raise ValueError(
                f"Embedding Error: {response.text}"
            )

        return [
            item["embedding"]
            for item in response.json()["data"]
        ]

# =========================================================
# INITIALIZE CHROMADB
# =========================================================
@st.cache_resource
def init_chromadb():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

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
# INITIALIZE LLM
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
# RAG FUNCTION
# =========================================================
def get_rag_response(query, n_results=3):

    try:

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        docs = results.get("documents", [[]])[0]

        if not docs:
            return "❌ No relevant information found."

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

    st.markdown("## 🤖 AI HR Assistant")

    st.markdown("""
Your intelligent company knowledge assistant powered by:

- Semantic Search
- ChromaDB
- OpenRouter LLMs
- Retrieval-Augmented Generation
""")

    st.divider()

    st.metric(
        "📄 Indexed Documents",
        collection.count()
    )

    st.metric(
        "💬 Chat Messages",
        len(st.session_state.messages)
    )

    st.divider()

    st.markdown("### 🧠 Capabilities")

    st.markdown("""
- PTO & Vacation Policies  
- Remote Work Guidelines  
- Employee Benefits  
- HR Information  
- Company Policies  
""")

    st.divider()

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown(
    '<div class="main-title">🤖 Company Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Semantic Search + AI-powered RAG System</div>',
    unsafe_allow_html=True
)

# =========================================================
# WELCOME MESSAGE
# =========================================================
if len(st.session_state.messages) == 0:

    st.markdown("""
    <div class="welcome-box">
    <h3>👋 Welcome!</h3>

    I can help you with:

    ✅ PTO & Vacation Policies  
    ✅ Remote Work Rules  
    ✅ HR Benefits  
    ✅ Company Guidelines  
    ✅ Employee Policies  

    Ask any question below to get started.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================================================
# CHAT INPUT
# =========================================================
if prompt := st.chat_input(
    "Ask a question about company policies..."
):

    # User Message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    # Assistant Response
    with st.chat_message("assistant"):

        with st.spinner("🔍 Searching knowledge base..."):

            response = get_rag_response(prompt)

        st.write(response)

    # Save Response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
