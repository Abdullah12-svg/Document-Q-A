
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from loaders import load_document
from splitter import split_documents
from vectorstore import create_vectorstore
from qa import create_qa_chain


load_dotenv()


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "ask_question" not in st.session_state:
    st.session_state.ask_question = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

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


# --------------------------------------------------
# HEADER
# --------------------------------------------------

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


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

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


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

if uploaded_file:

    if st.session_state.file_name != uploaded_file.name:

        try:

            with st.spinner(
                "🔄 Processing document..."
            ):

                file_extension = os.path.splitext(
                    uploaded_file.name
                )[1]

                # Check for empty file
                if uploaded_file.size == 0:
                    raise ValueError(
                        "The uploaded file is empty."
                    )

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=file_extension
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_file_path = temp_file.name

                # Load document
                try:

                    documents = load_document(
                        temp_file_path
                    )

                finally:

                    if os.path.exists(temp_file_path):

                        os.unlink(
                            temp_file_path
                        )

                # Check loaded documents
                if not documents:

                    raise ValueError(
                        "No readable content was found "
                        "in this document."
                    )

                # Preserve original filename
                for document in documents:

                    document.metadata["source"] = (
                        uploaded_file.name
                    )

                # Split document
                chunks = split_documents(
                    documents
                )

                # Check chunks
                if not chunks:

                    raise ValueError(
                        "The document could not be "
                        "split into readable chunks."
                    )

                # Create unique Chroma collection
                collection_name = (
                    "document_"
                    + uploaded_file.name
                    .replace(".", "_")
                    .replace(" ", "_")
                )

                # Create vector store
                vectorstore = create_vectorstore(
                    chunks,
                    collection_name
                )

                # Create QA chain
                st.session_state.ask_question = (
                    create_qa_chain(
                        vectorstore
                    )
                )

                # Save state
                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.session_state.chunks_count = (
                    len(chunks)
                )

                # Clear previous conversation
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

            # Reset state after error
            st.session_state.ask_question = None

            st.session_state.file_name = None

            st.session_state.chunks_count = 0

            st.session_state.messages = []


# --------------------------------------------------
# DOCUMENT STATUS
# --------------------------------------------------

if st.session_state.ask_question:

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


    # --------------------------------------------------
    # QUESTION AREA
    # --------------------------------------------------

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Your question",
        placeholder="What is this document about?",
        label_visibility="collapsed"
    )

    ask_button = st.button(
        "🔍 Ask Question",
        use_container_width=True
    )


    # --------------------------------------------------
    # ASK QUESTION
    # --------------------------------------------------

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "🤔 Searching the document..."
                ):

                    result = (
                        st.session_state.ask_question(
                            question
                        )
                    )

                # Store conversation
                st.session_state.messages.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"]
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


    # --------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------

    if st.session_state.messages:

        st.subheader("💡 Answers")

        for message in reversed(
            st.session_state.messages
        ):

            st.markdown(
                f"**🙋 You:** "
                f"{message['question']}"
            )

            st.markdown(
                f"**🤖 Nova:** "
                f"{message['answer']}"
            )


            # --------------------------------------------------
            # SOURCES
            # --------------------------------------------------

            if message["sources"]:

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


                        # PDF page number
                        if page is not None:

                            st.write(
                                f"📑 **Page:** "
                                f"{page + 1}"
                            )


                        # Retrieved chunk
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


            st.divider()


# --------------------------------------------------
# EMPTY STATE
# --------------------------------------------------

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

