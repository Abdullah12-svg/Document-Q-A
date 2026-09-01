
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from vectorstore import (
    create_retriever,
    get_all_documents
)


# --------------------------------------------------
# EXTRACT CLEAN TEXT FROM GEMINI RESPONSE
# --------------------------------------------------

def extract_text(response):

    content = response.content

    # Gemini returns normal string
    if isinstance(content, str):

        return content.strip()

    # Gemini returns a list of content blocks
    if isinstance(content, list):

        text_parts = []

        for item in content:

            # Dictionary content block
            if isinstance(item, dict):

                text = item.get("text")

                if text:

                    text_parts.append(
                        text
                    )

            # Plain string
            elif isinstance(item, str):

                text_parts.append(
                    item
                )

        if text_parts:

            return "\n".join(
                text_parts
            ).strip()

    # Fallback
    return str(content).strip()


# --------------------------------------------------
# CREATE QA CHAIN
# --------------------------------------------------

def create_qa_chain(vectorstore):

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0
    )


    # --------------------------------------------------
    # NORMAL RAG RETRIEVER
    # --------------------------------------------------

    retriever = create_retriever(
        vectorstore
    )


    # --------------------------------------------------
    # NORMAL QUESTION PROMPT
    # --------------------------------------------------

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

5. Give a clear and concise answer.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""
    )


    # --------------------------------------------------
    # DOCUMENT OVERVIEW PROMPT
    # --------------------------------------------------

    overview_prompt = ChatPromptTemplate.from_template(
        """
You are a document summarization assistant.

The user wants to know what the document is about.

You have been given ALL available document chunks.

IMPORTANT RULES:

1. Read ALL of the provided document information.
2. Consider every employee, row, record, or section.
3. Do NOT focus only on the first or most relevant entry.
4. Identify the overall subject of the document.
5. Explain what type of information the document contains.
6. If the document contains multiple records, describe
   the overall dataset rather than focusing on one record.
7. Do NOT use outside knowledge.
8. Do NOT invent information.
9. Keep the answer concise.

DOCUMENT:

{context}

USER QUESTION:

{question}

ANSWER:
"""
    )


    # --------------------------------------------------
    # OVERVIEW QUESTION DETECTION
    # --------------------------------------------------

    overview_questions = [

        "what is this document about",

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

    ]


    # --------------------------------------------------
    # ASK QUESTION
    # --------------------------------------------------

    def ask_question(question):

        question_clean = question.strip()

        question_lower = question_clean.lower()


        # ---------------------------------------------
        # CHECK QUESTION TYPE
        # ---------------------------------------------

        is_overview_question = any(

            phrase in question_lower

            for phrase in overview_questions

        )


        # ---------------------------------------------
        # OVERVIEW QUESTION
        # ---------------------------------------------

        if is_overview_question:

            all_documents = get_all_documents(
                vectorstore
            )

            if not all_documents:

                return {
                    "answer": (
                        "I don't know based on the "
                        "provided document."
                    ),
                    "sources": []
                }


            # Create context from ALL chunks

            context = "\n\n".join(

                item["content"]

                for item in all_documents

                if item.get("content")

            )


            # Create overview prompt

            messages = overview_prompt.invoke(
                {
                    "context": context,
                    "question": question_clean
                }
            )


            # Sources

            sources = [

                {
                    "source": item[
                        "metadata"
                    ].get(
                        "source",
                        "Unknown"
                    ),

                    "page": item[
                        "metadata"
                    ].get(
                        "page"
                    ),

                    "content": item[
                        "content"
                    ]

                }

                for item in all_documents[:5]

                if item.get("content")

            ]


        # ---------------------------------------------
        # NORMAL QUESTION
        # ---------------------------------------------

        else:

            documents = retriever.invoke(
                question_clean
            )


            if not documents:

                return {
                    "answer": (
                        "I don't know based on the "
                        "provided document."
                    ),
                    "sources": []
                }


            # Create context

            context = "\n\n".join(

                document.page_content

                for document in documents

            )


            # Create normal QA prompt

            messages = qa_prompt.invoke(
                {
                    "context": context,
                    "question": question_clean
                }
            )


            # Sources

            sources = [

                {
                    "source": document.metadata.get(
                        "source",
                        "Unknown"
                    ),

                    "page": document.metadata.get(
                        "page"
                    ),

                    "content": document.page_content

                }

                for document in documents

            ]


        # ---------------------------------------------
        # CALL GEMINI
        # ---------------------------------------------

        response = llm.invoke(
            messages
        )


        # ---------------------------------------------
        # EXTRACT CLEAN ANSWER
        # ---------------------------------------------

        answer = extract_text(
            response
        )


        # ---------------------------------------------
        # RETURN RESULT
        # ---------------------------------------------

        return {

            "answer": answer,

            "sources": sources

        }


    # --------------------------------------------------
    # RETURN QA FUNCTION
    # --------------------------------------------------

    return ask_question
