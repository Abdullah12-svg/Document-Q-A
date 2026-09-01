
import os
import tempfile
import hashlib

import streamlit as st
from dotenv import load_dotenv

from loaders import load_document
from splitter import split_documents
from vectorstore import create_vectorstore
from qa import create_qa_chain


# ==================================================
# LOAD ENVIRONMENT
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

if "file_names" not in st.session_state:
    st.session_state.file_names = []

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
    'Ask questions across multiple PDF, CSV, and TXT '
    'documents using Retrieval-Augmented Generation (RAG).'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True
    )

    st.divider()

    st.header("ℹ️ Supported Files")

    st.write("📕 PDF")
    st.write("📊 CSV")
    st.write("📝 TXT")

    st.divider()

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

if uploaded_files:

    # --------------------------------------------------
    # CURRENT FILE SET
    # --------------------------------------------------

    current_file_names = sorted(
        file.name
        for file in uploaded_files
    )

    previous_file_names = sorted(
        st.session_state.file_names
    )


    # --------------------------------------------------
    # PROCESS ONLY WHEN FILE SET CHANGES
    # --------------------------------------------------

    if current_file_names != previous_file_names:

        try:

            with st.spinner(
                "🔄 Processing documents..."
            ):

                all_chunks = []


                # ==================================================
                # PROCESS EVERY UPLOADED FILE
                # ==================================================

                for uploaded_file in uploaded_files:

                    file_extension = os.path.splitext(
                        uploaded_file.name
                    )[1]


                    # --------------------------------------------------
                    # EMPTY FILE CHECK
                    # --------------------------------------------------

                    if uploaded_file.size == 0:

                        raise ValueError(
                            f"{uploaded_file.name} is empty."
                        )


                    # --------------------------------------------------
                    # CREATE TEMPORARY FILE
                    # --------------------------------------------------

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=file_extension
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getvalue()
                        )

                        temp_file_path = (
                            temp_file.name
                        )


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
                            f"No readable content found "
                            f"in {uploaded_file.name}."
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


                    # --------------------------------------------------
                    # CHECK CHUNKS
                    # --------------------------------------------------

                    if not chunks:

                        raise ValueError(
                            f"{uploaded_file.name} could "
                            f"not be split into chunks."
                        )


                    # --------------------------------------------------
                    # ADD CHUNKS TO GLOBAL LIST
                    # --------------------------------------------------

                    all_chunks.extend(
                        chunks
                    )


                # ==================================================
                # CHECK TOTAL CHUNKS
                # ==================================================

                if not all_chunks:

                    raise ValueError(
                        "No readable chunks were created "
                        "from the uploaded files."
                    )


                # ==================================================
                # CREATE UNIQUE COLLECTION
                # ==================================================

                files_key = "|".join(
                    current_file_names
                )

                collection_hash = hashlib.md5(
                    files_key.encode()
                ).hexdigest()[:12]

                collection_name = (
                    "multi_document_"
                    + collection_hash
                )


                # ==================================================
                # CREATE VECTOR STORE
                # ==================================================

                vectorstore = create_vectorstore(
                    all_chunks,
                    collection_name
                )


                # ==================================================
                # CREATE QA CHAIN
                # ==================================================

                st.session_state.ask_question = (
                    create_qa_chain(
                        vectorstore
                    )
                )


                # ==================================================
                # SAVE SESSION STATE
                # ==================================================

                st.session_state.file_names = (
                    current_file_names
                )

                st.session_state.chunks_count = (
                    len(all_chunks)
                )


                # --------------------------------------------------
                # RESET CONVERSATION FOR NEW FILE SET
                # --------------------------------------------------

                st.session_state.messages = []


            # ==================================================
            # SUCCESS
            # ==================================================

            st.success(
                f"✅ {len(uploaded_files)} document(s) "
                "processed successfully!"
            )


        # ==================================================
        # ERROR HANDLING
        # ==================================================

        except Exception as e:

            st.error(
                "❌ Could not process the documents."
            )

            st.info(
                f"Details: {str(e)}"
            )


            st.session_state.ask_question = None

            st.session_state.file_names = []

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


    # --------------------------------------------------
    # DOCUMENT LIST
    # --------------------------------------------------

    with col1:

        st.write(
            "📄 **Documents**"
        )

        for file_name in (
            st.session_state.file_names
        ):

            st.write(
                f"• {file_name}"
            )


    # --------------------------------------------------
    # CHUNK COUNT
    # --------------------------------------------------

    with col2:

        st.write(
            "🧩 **Total Chunks**"
        )

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

    for message in (
        st.session_state.messages
    ):


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

        with st.chat_message("assistant"):

            st.write(
                message["answer"]
            )


            # ==================================================
            # SOURCES
            # ==================================================

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

                        row = source.get(
                            "row"
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
                        # CSV ROW
                        # --------------------------------------------------

                        if row is not None:

                            st.write(
                                f"📊 **Row:** "
                                f"{row}"
                            )


                        # --------------------------------------------------
                        # RETRIEVED CONTENT
                        # --------------------------------------------------

                        st.text_area(
                            "Retrieved content",
                            content,
                            height=150,
                            key=(
                                f"history_source_"
                                f"{index}_"
                                f"{id(message)}"
                            )
                        )


    # ==================================================
    # CHAT INPUT
    # ==================================================

    question = st.chat_input(
        "Ask something about your documents..."
    )


    # ==================================================
    # PROCESS QUESTION
    # ==================================================

    if question:

        # --------------------------------------------------
        # USER MESSAGE
        # --------------------------------------------------

        with st.chat_message("user"):

            st.write(
                question
            )


        try:

            # --------------------------------------------------
            # ASSISTANT RESPONSE
            # --------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "🤔 Searching your documents..."
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


                # ==================================================
                # DISPLAY SOURCES
                # ==================================================

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

                            row = source.get(
                                "row"
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
                            # CSV ROW
                            # --------------------------------------------------

                            if row is not None:

                                st.write(
                                    f"📊 **Row:** "
                                    f"{row}"
                                )


                            # --------------------------------------------------
                            # CONTENT
                            # --------------------------------------------------

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


                # ==================================================
                # SAVE MESSAGE
                # ==================================================

                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    }
                )


        # ==================================================
        # QUESTION ERROR
        # ==================================================

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
        "👈 Upload one or more PDF, CSV, or TXT "
        "files from the sidebar to get started."
    )


    col1, col2, col3 = st.columns(3)


    # ==================================================
    # PDF
    # ==================================================

    with col1:

        st.markdown(
            "### 📕 PDF"
        )

        st.write(
            "Ask questions across reports, "
            "notes, books and documents."
        )


    # ==================================================
    # CSV
    # ==================================================

    with col2:

        st.markdown(
            "### 📊 CSV"
        )

        st.write(
            "Search information across "
            "multiple datasets."
        )


    # ==================================================
    # TXT
    # ==================================================

    with col3:

        st.markdown(
            "### 📝 TXT"
        )

        st.write(
            "Ask questions across multiple "
            "text files."
        )
