import os
import time
import tempfile

import pandas as pd
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq

from sentence_transformers import CrossEncoder


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HybridRAG | AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .source-card {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 8px;
    }

    .status-box {
        padding: 10px;
        border-radius: 10px;
        background-color: rgba(100, 100, 100, 0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY was not found. "
        "Please add it to your .env file."
    )
    st.stop()


# =========================================================
# CACHED MODELS
# =========================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


@st.cache_resource
def load_llm():

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=GROQ_API_KEY
    )


# Load models once
embeddings = load_embeddings()
reranker = load_reranker()
llm = load_llm()


# =========================================================
# SESSION STATE
# =========================================================

if "rag_data" not in st.session_state:
    st.session_state.rag_data = None

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_analytics" not in st.session_state:
    st.session_state.last_analytics = None


# =========================================================
# RESET FUNCTION
# =========================================================

def reset_application():

    st.session_state.rag_data = None
    st.session_state.file_signature = None
    st.session_state.messages = []
    st.session_state.last_analytics = None


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📚 HybridRAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Multi-Document AI Research Assistant powered by '
    'Hybrid Retrieval and Cross-Encoder Re-ranking'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📄 Document Center")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents."
    )

    st.divider()

    # -----------------------------------------------------
    # DOCUMENT LIST
    # -----------------------------------------------------

    if uploaded_files:

        st.subheader("Your Documents")

        for file in uploaded_files:

            st.markdown(
                f"📄 **{file.name}**"
            )

        st.caption(
            f"{len(uploaded_files)} document(s) selected"
        )

    else:

        st.info(
            "Upload PDFs to start."
        )

    st.divider()

    # -----------------------------------------------------
    # SYSTEM STATUS
    # -----------------------------------------------------

    st.subheader("⚙️ System Status")

    if st.session_state.rag_data:

        st.success("🟢 RAG System Ready")

        total_chunks = len(
            st.session_state.rag_data["chunks"]
        )

        st.metric(
            "Indexed Chunks",
            total_chunks
        )

    else:

        st.warning(
            "🟡 Waiting for documents"
        )

    st.divider()

    # -----------------------------------------------------
    # PIPELINE
    # -----------------------------------------------------

    st.subheader("🔎 RAG Pipeline")

    st.markdown(
        """
        **1. BM25**  
        Keyword retrieval

        **2. FAISS**  
        Semantic retrieval

        **3. Hybrid Retrieval**  
        BM25 + FAISS

        **4. Cross-Encoder**  
        Re-ranking

        **5. Groq LLM**  
        Answer generation
        """
    )

    st.divider()

    # -----------------------------------------------------
    # RESET BUTTON
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Chat & Documents",
        use_container_width=True
    ):

        reset_application()

        st.rerun()

    st.caption(
        "Your uploaded documents are processed "
        "for the current session."
    )


# =========================================================
# PROCESS DOCUMENTS
# =========================================================

if uploaded_files:

    current_signature = tuple(
        (
            file.name,
            len(file.getvalue())
        )
        for file in uploaded_files
    )

    # Process only when files change
    if current_signature != st.session_state.file_signature:

        with st.spinner(
            "🔄 Processing your documents..."
        ):

            documents = []

            for uploaded_file in uploaded_files:

                temp_pdf_path = None

                try:

                    # Temporary PDF
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getvalue()
                        )

                        temp_pdf_path = temp_file.name

                    # Load PDF
                    loader = PyPDFLoader(
                        temp_pdf_path
                    )

                    pdf_documents = loader.load()

                    # Preserve filename
                    for doc in pdf_documents:

                        doc.metadata["source"] = (
                            uploaded_file.name
                        )

                    documents.extend(
                        pdf_documents
                    )

                except Exception as e:

                    st.error(
                        f"Could not process "
                        f"{uploaded_file.name}: {e}"
                    )

                finally:

                    if (
                        temp_pdf_path
                        and os.path.exists(
                            temp_pdf_path
                        )
                    ):

                        os.remove(
                            temp_pdf_path
                        )

            # -------------------------------------------------
            # CHUNKING
            # -------------------------------------------------

            text_splitter = (
                RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
            )

            chunks = (
                text_splitter.split_documents(
                    documents
                )
            )

            # -------------------------------------------------
            # FAISS
            # -------------------------------------------------

            vectordb = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            dense_retriever = (
                vectordb.as_retriever(
                    search_kwargs={"k": 5}
                )
            )

            # -------------------------------------------------
            # BM25
            # -------------------------------------------------

            bm25_retriever = (
                BM25Retriever.from_documents(
                    chunks
                )
            )

            bm25_retriever.k = 5

            # -------------------------------------------------
            # HYBRID RETRIEVER
            # -------------------------------------------------

            ensemble_retriever = (
                EnsembleRetriever(
                    retrievers=[
                        bm25_retriever,
                        dense_retriever
                    ],
                    weights=[
                        0.4,
                        0.6
                    ]
                )
            )

            # -------------------------------------------------
            # SAVE RAG SYSTEM
            # -------------------------------------------------

            st.session_state.rag_data = {

                "documents": documents,

                "chunks": chunks,

                "vectordb": vectordb,

                "bm25": bm25_retriever,

                "dense": dense_retriever,

                "hybrid": ensemble_retriever
            }

            st.session_state.file_signature = (
                current_signature
            )

            # New documents = new conversation
            st.session_state.messages = []

            st.session_state.last_analytics = None

        st.success(
            f"✅ Indexed {len(documents)} pages "
            f"into {len(chunks)} chunks."
        )


# =========================================================
# NO DOCUMENT STATE
# =========================================================

if not uploaded_files:

    st.markdown("### 👋 Welcome to HybridRAG")

    st.write(
        "Upload one or more PDF documents from the "
        "sidebar and ask questions using natural language."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "📄 **Multi-Document**\n\n"
            "Ask questions across multiple PDFs."
        )

    with col2:
        st.info(
            "🔎 **Hybrid Retrieval**\n\n"
            "Combines BM25 and semantic search."
        )

    with col3:
        st.info(
            "🎯 **Re-ranking**\n\n"
            "Cross-Encoder selects the most relevant context."
        )

    st.stop()


# =========================================================
# SAFETY CHECK
# =========================================================

if st.session_state.rag_data is None:

    st.warning(
        "Please wait while your documents are indexed."
    )

    st.stop()


# =========================================================
# GET RETRIEVERS
# =========================================================

bm25_retriever = (
    st.session_state.rag_data["bm25"]
)

dense_retriever = (
    st.session_state.rag_data["dense"]
)

ensemble_retriever = (
    st.session_state.rag_data["hybrid"]
)


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER QUESTION
# =========================================================

query = st.chat_input(
    "Ask a question about your documents...",
    key="main_chat_input"
)


# =========================================================
# QUESTION PROCESSING
# =========================================================

if query:

    # -----------------------------------------------------
    # SHOW USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    total_start = time.perf_counter()

    # -----------------------------------------------------
    # BM25 RETRIEVAL
    # -----------------------------------------------------

    start = time.perf_counter()

    try:

        bm25_docs = (
            bm25_retriever.invoke(query)
        )

        bm25_latency = (
            time.perf_counter() - start
        )

    except Exception:

        bm25_docs = []

        bm25_latency = (
            time.perf_counter() - start
        )

    # -----------------------------------------------------
    # DENSE RETRIEVAL
    # -----------------------------------------------------

    start = time.perf_counter()

    try:

        dense_docs = (
            dense_retriever.invoke(query)
        )

        dense_latency = (
            time.perf_counter() - start
        )

    except Exception:

        dense_docs = []

        dense_latency = (
            time.perf_counter() - start
        )

    # -----------------------------------------------------
    # HYBRID RETRIEVAL
    # -----------------------------------------------------

    start = time.perf_counter()

    try:

        hybrid_docs = (
            ensemble_retriever.invoke(query)
        )

        hybrid_latency = (
            time.perf_counter() - start
        )

    except Exception as e:

        st.error(
            f"Hybrid retrieval failed: {e}"
        )

        st.stop()

    # -----------------------------------------------------
    # CHECK RETRIEVAL
    # -----------------------------------------------------

    if not hybrid_docs:

        st.warning(
            "I could not retrieve relevant context "
            "from the uploaded documents."
        )

        st.stop()

    # -----------------------------------------------------
    # CROSS-ENCODER
    # -----------------------------------------------------

    pairs = [
        (
            query,
            doc.page_content
        )
        for doc in hybrid_docs
    ]

    start = time.perf_counter()

    scores = reranker.predict(
        pairs
    )

    reranker_latency = (
        time.perf_counter() - start
    )

    # -----------------------------------------------------
    # RANK DOCUMENTS
    # -----------------------------------------------------

    ranked_docs = sorted(
        zip(
            scores,
            hybrid_docs
        ),
        key=lambda item: item[0],
        reverse=True
    )

    top_docs = [
        doc
        for score, doc
        in ranked_docs[:5]
    ]

    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in top_docs
    )

    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are HybridRAG, a document question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
"I could not find the answer in the provided documents."
4. Give a clear and useful answer.
5. Keep the answer concise unless the question requires detail.

Retrieved Context:
{context}

User Question:
{query}

Answer:
"""

    # -----------------------------------------------------
    # LLM
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤔 Generating answer..."
        ):

            llm_start = time.perf_counter()

            try:

                response = llm.invoke(
                    prompt
                )

                answer = response.content

            except Exception as e:

                answer = (
                    "Sorry, I could not generate "
                    f"an answer. Error: {e}"
                )

            llm_latency = (
                time.perf_counter()
                - llm_start
            )

        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        st.markdown(answer)

        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        st.markdown("### 📚 Sources")

        for index, doc in enumerate(
            top_docs
        ):

            source = doc.metadata.get(
                "source",
                "Unknown document"
            )

            page = doc.metadata.get(
                "page",
                None
            )

            if page is not None:

                page_text = (
                    f"Page {page + 1}"
                )

            else:

                page_text = (
                    "Page unavailable"
                )

            with st.expander(
                f"📄 {source} • {page_text}"
            ):

                st.caption(
                    f"Retrieved Source {index + 1}"
                )

                st.markdown(
                    "**Relevant Context**"
                )

                st.write(
                    doc.page_content
                )

        # -------------------------------------------------
        # RETRIEVAL DETAILS
        # -------------------------------------------------

        with st.expander(
            "🔍 View Retrieval Details"
        ):

            detail_col1, detail_col2 = (
                st.columns(2)
            )

            with detail_col1:

                st.metric(
                    "BM25 Candidates",
                    len(bm25_docs)
                )

                st.metric(
                    "Dense Candidates",
                    len(dense_docs)
                )

            with detail_col2:

                st.metric(
                    "Hybrid Candidates",
                    len(hybrid_docs)
                )

                st.metric(
                    "Final Context",
                    len(top_docs)
                )

        # -------------------------------------------------
        # ANALYTICS
        # -------------------------------------------------

        hybrid_reranker_latency = (
            hybrid_latency
            + reranker_latency
        )

        total_response_time = (
            time.perf_counter()
            - total_start
        )

        analytics_data = {

            "Method": [
                "BM25",
                "Dense / FAISS",
                "Hybrid",
                "Hybrid + Reranker"
            ],

            "Latency (seconds)": [

                round(
                    bm25_latency,
                    4
                ),

                round(
                    dense_latency,
                    4
                ),

                round(
                    hybrid_latency,
                    4
                ),

                round(
                    hybrid_reranker_latency,
                    4
                )
            ]
        }

        analytics_df = pd.DataFrame(
            analytics_data
        )

        st.session_state.last_analytics = (
            analytics_df
        )

        with st.expander(
            "📊 Retrieval Analytics"
        ):

            metric1, metric2, metric3 = (
                st.columns(3)
            )

            with metric1:

                st.metric(
                    "Total Response",
                    f"{total_response_time:.2f}s"
                )

            with metric2:

                st.metric(
                    "LLM Generation",
                    f"{llm_latency:.2f}s"
                )

            with metric3:

                st.metric(
                    "Final Chunks",
                    len(top_docs)
                )

            st.markdown(
                "#### Retrieval Latency"
            )

            st.dataframe(
                analytics_df,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                analytics_df.set_index(
                    "Method"
                )[
                    "Latency (seconds)"
                ]
            )

            st.caption(
                "Latency values are measured for the "
                "current question and can vary by machine "
                "and network conditions."
            )

    # -----------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )