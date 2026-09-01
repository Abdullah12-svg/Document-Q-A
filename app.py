
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from loaders import load_document
from splitter import split_documents
from vectorstore import create_vectorstore
from qa import create_qa_chain


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# SESSION STATE
# ==================================================

if "ask_question" not in st.session_state:
    st.session_state.ask_question = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .status-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">📚 Document Q&A</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your PDF, CSV, or TXT documents '
    'using Retrieval-Augmented Generation (RAG).'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["pdf", "csv", "txt"]
    )

    st.divider()

    st.header("ℹ️ Supported Files")

    st.write("📕 PDF")
    st.write("📊 CSV")
    st.write("📝 TXT")

    st.divider()

    # --------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------

    st.header("💬 Conversation")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==================================================
# DOCUMENT PROCESSING
# ==================================================

if uploaded_file:

    # Process only when a new document is uploaded
    if st.session_state.file_name != uploaded_file.name:

        try:

            with st.spinner(
                "🔄 Processing document..."
            ):

                # --------------------------------------------------
                # FILE EXTENSION
                # --------------------------------------------------

                file_extension = os.path.splitext(
                    uploaded_file.name
                )[1]

                # --------------------------------------------------
                # EMPTY FILE CHECK
                # --------------------------------------------------

                if uploaded_file.size == 0:

                    raise ValueError(
                        "The uploaded file is empty."
                    )

                # --------------------------------------------------
                # SAVE TEMPORARY FILE
                # --------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=file_extension
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_file_path = temp_file.name

                # --------------------------------------------------
                # LOAD DOCUMENT
                # --------------------------------------------------

                try:

                    documents = load_document(
                        temp_file_path
                    )

                finally:

                    if os.path.exists(
                        temp_file_path
                    ):

                        os.unlink(
                            temp_file_path
                        )

                # --------------------------------------------------
                # CHECK DOCUMENT
                # --------------------------------------------------

                if not documents:

                    raise ValueError(
                        "No readable content was found "
                        "in this document."
                    )

                # --------------------------------------------------
                # PRESERVE ORIGINAL FILE NAME
                # --------------------------------------------------

                for document in documents:

                    document.metadata["source"] = (
                        uploaded_file.name
                    )

                # --------------------------------------------------
                # SPLIT DOCUMENT
                # --------------------------------------------------

                chunks = split_documents(
                    documents
                )

                if not chunks:

                    raise ValueError(
                        "The document could not be "
                        "split into readable chunks."
                    )

                # --------------------------------------------------
                # CREATE COLLECTION NAME
                # --------------------------------------------------

                collection_name = (
                    "document_"
                    + uploaded_file.name
                    .replace(".", "_")
                    .replace(" ", "_")
                )

                # --------------------------------------------------
                # CREATE VECTOR STORE
                # --------------------------------------------------

                vectorstore = create_vectorstore(
                    chunks,
                    collection_name
                )

                # --------------------------------------------------
                # CREATE QA CHAIN
                # --------------------------------------------------

                st.session_state.ask_question = (
                    create_qa_chain(
                        vectorstore
                    )
                )

                # --------------------------------------------------
                # SAVE DOCUMENT STATE
                # --------------------------------------------------

                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.session_state.chunks_count = (
                    len(chunks)
                )

                # New document = new conversation
                st.session_state.messages = []

            st.success(
                f"✅ {uploaded_file.name} "
                "processed successfully!"
            )

        except Exception as e:

            st.error(
                "❌ Could not process this document."
            )

            st.info(
                f"Details: {str(e)}"
            )

            # --------------------------------------------------
            # RESET STATE
            # --------------------------------------------------

            st.session_state.ask_question = None

            st.session_state.file_name = None

            st.session_state.chunks_count = 0

            st.session_state.messages = []


# ==================================================
# DOCUMENT READY
# ==================================================

if st.session_state.ask_question:

    # ==================================================
    # DOCUMENT STATUS
    # ==================================================

    st.markdown(
        '<div class="status-box">',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("📄 **Document**")

        st.write(
            st.session_state.file_name
        )

    with col2:

        st.write("🧩 **Chunks**")

        st.write(
            st.session_state.chunks_count
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ==================================================
    # CHAT HISTORY
    # ==================================================

    for message in st.session_state.messages:

        # --------------------------------------------------
        # USER MESSAGE
        # --------------------------------------------------

        with st.chat_message("user"):

            st.write(
                message["question"]
            )


        # --------------------------------------------------
        # ASSISTANT MESSAGE
        # --------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            st.write(
                message["answer"]
            )

            # --------------------------------------------------
            # SOURCES
            # --------------------------------------------------

            if message.get("sources"):

                with st.expander(
                    "📚 View Sources"
                ):

                    for index, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        source_name = os.path.basename(
                            source.get(
                                "source",
                                "Unknown"
                            )
                        )

                        page = source.get(
                            "page"
                        )

                        content = source.get(
                            "content",
                            ""
                        )

                        st.markdown(
                            f"### Source {index}"
                        )

                        st.write(
                            f"📄 **File:** "
                            f"{source_name}"
                        )

                        # --------------------------------------------------
                        # PDF PAGE
                        # --------------------------------------------------

                        if page is not None:

                            st.write(
                                f"📑 **Page:** "
                                f"{page + 1}"
                            )

                        # --------------------------------------------------
                        # RETRIEVED CONTENT
                        # --------------------------------------------------

                        st.text_area(
                            "Retrieved content",
                            content,
                            height=150,
                            key=(
                                f"source_"
                                f"{index}_"
                                f"{id(message)}"
                            )
                        )


    # ==================================================
    # CHAT INPUT
    # ==================================================

    question = st.chat_input(
        "Ask something about your document..."
    )


    # ==================================================
    # PROCESS QUESTION
    # ==================================================

    if question:

        # --------------------------------------------------
        # DISPLAY USER MESSAGE IMMEDIATELY
        # --------------------------------------------------

        with st.chat_message("user"):

            st.write(
                question
            )

        try:

            # --------------------------------------------------
            # GENERATE ANSWER
            # --------------------------------------------------

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "🤔 Searching the document..."
                ):

                    result = (
                        st.session_state.ask_question(
                            question,
                            st.session_state.messages
                        )
                    )

                answer = result["answer"]

                sources = result["sources"]

                # --------------------------------------------------
                # DISPLAY ANSWER
                # --------------------------------------------------

                st.write(
                    answer
                )

                # --------------------------------------------------
                # DISPLAY SOURCES
                # --------------------------------------------------

                if sources:

                    with st.expander(
                        "📚 View Sources"
                    ):

                        for index, source in enumerate(
                            sources,
                            start=1
                        ):

                            source_name = os.path.basename(
                                source.get(
                                    "source",
                                    "Unknown"
                                )
                            )

                            page = source.get(
                                "page"
                            )

                            content = source.get(
                                "content",
                                ""
                            )

                            st.markdown(
                                f"### Source {index}"
                            )

                            st.write(
                                f"📄 **File:** "
                                f"{source_name}"
                            )

                            if page is not None:

                                st.write(
                                    f"📑 **Page:** "
                                    f"{page + 1}"
                                )

                            st.text_area(
                                "Retrieved content",
                                content,
                                height=150,
                                key=(
                                    f"new_source_"
                                    f"{index}_"
                                    f"{len(st.session_state.messages)}"
                                )
                            )

                # --------------------------------------------------
                # SAVE MESSAGE
                # --------------------------------------------------

                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    }
                )

        except Exception as e:

            st.error(
                "❌ Something went wrong "
                "while generating the answer."
            )

            st.info(
                f"Details: {str(e)}"
            )


# ==================================================
# EMPTY STATE
# ==================================================

else:

    st.info(
        "👈 Upload a PDF, CSV, or TXT file "
        "from the sidebar to get started."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 📕 PDF")

        st.write(
            "Ask questions about reports, "
            "notes, books and documents."
        )

    with col2:

        st.markdown("### 📊 CSV")

        st.write(
            "Search information stored "
            "in tabular data."
        )

    with col3:

        st.markdown("### 📝 TXT")

        st.write(
            "Ask questions about plain "
            "text files."
        )
