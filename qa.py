
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from vectorstore import (
    create_retriever,
    get_all_documents
)


# ==================================================
# EXTRACT CLEAN TEXT FROM GEMINI RESPONSE
# ==================================================

def extract_text(response):

    content = response.content

    # Gemini returns normal string
    if isinstance(content, str):

        return content.strip()

    # Gemini returns content blocks
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get("text")

                if text:

                    text_parts.append(
                        text
                    )

            elif isinstance(item, str):

                text_parts.append(
                    item
                )

        if text_parts:

            return "\n".join(
                text_parts
            ).strip()

    return str(content).strip()


# ==================================================
# REMOVE DUPLICATE DOCUMENTS
# ==================================================

def remove_duplicate_documents(
    documents
):

    unique_documents = []

    seen = set()

    for document in documents:

        content = (
            document.page_content
            .strip()
        )

        metadata = (
            document.metadata
            or {}
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page"
        )

        row = metadata.get(
            "row"
        )

        document_id = (
            content,
            source,
            page,
            row
        )

        if document_id in seen:

            continue

        seen.add(
            document_id
        )

        unique_documents.append(
            document
        )

    return unique_documents


# ==================================================
# LIMIT CONTEXT SIZE
# ==================================================

def build_context(
    documents,
    max_characters=12000
):

    context_parts = []

    total_characters = 0

    for document in documents:

        content = (
            document.page_content
            .strip()
        )

        if not content:

            continue

        remaining = (
            max_characters
            - total_characters
        )

        if remaining <= 0:

            break

        if len(content) > remaining:

            content = content[
                :remaining
            ]

        context_parts.append(
            content
        )

        total_characters += (
            len(content)
        )

    return "\n\n".join(
        context_parts
    )


# ==================================================
# CREATE QA CHAIN
# ==================================================

def create_qa_chain(
    vectorstore
):

    # ==================================================
    # LLM
    # ==================================================

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0
    )


    # ==================================================
    # RETRIEVER
    # ==================================================

    retriever = create_retriever(
        vectorstore
    )


    # ==================================================
    # NORMAL QA PROMPT
    # ==================================================

    qa_prompt = ChatPromptTemplate.from_template(
        """
You are a document question-answering assistant.

Answer the user's question using ONLY the provided
document context.

IMPORTANT RULES:

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Do NOT invent information.
4. If the answer cannot be found in the context,
   say exactly:

"I don't know based on the provided document."

5. If multiple documents are provided, use information
   from any relevant document.
6. Give a clear and concise answer.
7. When useful, mention the document name containing
   the answer.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""
    )


    # ==================================================
    # OVERVIEW PROMPT
    # ==================================================

    overview_prompt = ChatPromptTemplate.from_template(
        """
You are a document summarization assistant.

The user wants to understand what the uploaded
documents are about.

You have been provided with information from the
uploaded documents.

IMPORTANT RULES:

1. Use ONLY the provided document information.
2. Do NOT use outside knowledge.
3. Do NOT invent information.
4. Consider information from ALL provided documents.
5. Do NOT focus on only one person, row, or chunk.
6. Identify the overall subject of the documents.
7. Explain the main types of information contained
   in the documents.
8. If there are multiple documents, mention the
   different documents and what each contains.
9. Keep the summary clear and reasonably concise.

DOCUMENT INFORMATION:

{context}

USER QUESTION:

{question}

ANSWER:
"""
    )


    # ==================================================
    # OVERVIEW QUESTIONS
    # ==================================================

    overview_questions = [

        "what is this document about",

        "what is the document about",

        "what does this document contain",

        "summarize this document",

        "summarise this document",

        "give me a summary",

        "what is this file about",

        "what does this file contain",

        "describe this document",

        "describe this file",

        "give me an overview",

        "provide an overview",

        "tell me about this document",

        "tell me about this file",

        "what are these documents about",

        "what do these documents contain",

        "summarize these documents",

        "summarise these documents",

        "give me an overview of these documents"

    ]


    # ==================================================
    # ASK QUESTION
    # ==================================================

    def ask_question(
        question,
        chat_history=None
    ):

        question_clean = (
            question.strip()
        )

        question_lower = (
            question_clean.lower()
        )


        # ==================================================
        # QUESTION TYPE
        # ==================================================

        is_overview_question = any(
            phrase in question_lower
            for phrase in overview_questions
        )


        # ==================================================
        # OVERVIEW
        # ==================================================

        if is_overview_question:

            all_documents = (
                get_all_documents(
                    vectorstore
                )
            )

            if not all_documents:

                return {
                    "answer": (
                        "I don't know based on "
                        "the provided document."
                    ),
                    "sources": []
                }


            # ------------------------------------------
            # Convert all chunks to temporary objects
            # ------------------------------------------

            class SimpleDocument:

                def __init__(
                    self,
                    content,
                    metadata
                ):

                    self.page_content = (
                        content
                    )

                    self.metadata = (
                        metadata
                    )


            overview_documents = [

                SimpleDocument(
                    item["content"],
                    item["metadata"]
                )

                for item in all_documents

                if item.get("content")
            ]


            # ------------------------------------------
            # Build overview context
            # ------------------------------------------

            context_documents = (
                remove_duplicate_documents(
                    overview_documents
                )
            )


            context = build_context(
                context_documents,
                max_characters=20000
            )


            if not context:

                return {
                    "answer": (
                        "I don't know based on "
                        "the provided document."
                    ),
                    "sources": []
                }


            messages = (
                overview_prompt.invoke(
                    {
                        "context": context,
                        "question": question_clean
                    }
                )
            )


            # ------------------------------------------
            # Sources
            # ------------------------------------------

            sources = []

            seen_sources = set()

            for document in context_documents:

                metadata = (
                    document.metadata
                    or {}
                )

                source = metadata.get(
                    "source",
                    "Unknown"
                )

                page = metadata.get(
                    "page"
                )

                row = metadata.get(
                    "row"
                )

                source_id = (
                    source,
                    page,
                    row
                )

                if source_id in seen_sources:

                    continue

                seen_sources.add(
                    source_id
                )

                sources.append(
                    {
                        "source": source,
                        "page": page,
                        "row": row,
                        "content": (
                            document.page_content
                        )
                    }
                )

                # Keep UI manageable
                if len(sources) >= 10:

                    break


        # ==================================================
        # NORMAL RAG QUESTION
        # ==================================================

        else:

            # ------------------------------------------
            # Retrieve relevant chunks
            # ------------------------------------------

            documents = retriever.invoke(
                question_clean
            )


            # ------------------------------------------
            # Remove duplicates
            # ------------------------------------------

            documents = (
                remove_duplicate_documents(
                    documents
                )
            )


            # ------------------------------------------
            # Maximum 5 retrieved chunks
            # ------------------------------------------

            documents = documents[:5]


            if not documents:

                return {
                    "answer": (
                        "I don't know based on "
                        "the provided document."
                    ),
                    "sources": []
                }


            # ------------------------------------------
            # Build context
            # ------------------------------------------

            context = build_context(
                documents,
                max_characters=12000
            )


            # ------------------------------------------
            # Prompt
            # ------------------------------------------

            messages = (
                qa_prompt.invoke(
                    {
                        "context": context,
                        "question": question_clean
                    }
                )
            )


            # ------------------------------------------
            # Sources
            # ------------------------------------------

            sources = [

                {
                    "source": document.metadata.get(
                        "source",
                        "Unknown"
                    ),

                    "page": document.metadata.get(
                        "page"
                    ),

                    "row": document.metadata.get(
                        "row"
                    ),

                    "content": document.page_content

                }

                for document in documents

            ]


        # ==================================================
        # CALL GEMINI
        # ==================================================

        response = llm.invoke(
            messages
        )


        # ==================================================
        # EXTRACT ANSWER
        # ==================================================

        answer = extract_text(
            response
        )


        # ==================================================
        # RETURN
        # ==================================================

        return {

            "answer": answer,

            "sources": sources

        }


    # ==================================================
    # RETURN FUNCTION
    # ==================================================

    return ask_question

