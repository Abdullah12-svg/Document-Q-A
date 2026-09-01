
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

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                text = item.get("text")

                if text:
                    text_parts.append(text)

            elif isinstance(item, str):

                text_parts.append(item)

        if text_parts:
            return "\n".join(text_parts).strip()

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
    # RETRIEVER
    # --------------------------------------------------

    retriever = create_retriever(
        vectorstore
    )

    # --------------------------------------------------
    # NORMAL QA PROMPT
    # --------------------------------------------------

    qa_prompt = ChatPromptTemplate.from_template(
        """
You are a document question-answering assistant.

Answer the user's question using ONLY the provided
document context.

IMPORTANT RULES:

1. Use ONLY the provided document context.
2. Do NOT use outside knowledge.
3. Do NOT invent information.
4. Use the conversation history only to understand
   references such as "he", "she", "it", "that person",
   "the previous one", etc.
5. The actual answer MUST come from the document context.
6. If the answer cannot be found in the document context,
   say exactly:

"I don't know based on the provided document."

7. Give a clear and concise answer.

CONVERSATION HISTORY:

{history}

DOCUMENT CONTEXT:

{context}

CURRENT USER QUESTION:

{question}

ANSWER:
"""
    )

    # --------------------------------------------------
    # OVERVIEW PROMPT
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
    # CONVERSATION-AWARE QUESTION REWRITE PROMPT
    # --------------------------------------------------

    rewrite_prompt = ChatPromptTemplate.from_template(
        """
You are a question rewriting assistant for a
document question-answering system.

Your job is to rewrite the user's current question
into a complete standalone question.

Use the conversation history to resolve references
such as:

- he
- she
- they
- it
- this person
- that person
- his
- her
- their
- the previous person
- the previous record

IMPORTANT RULES:

1. Do NOT answer the question.
2. ONLY rewrite the question.
3. Preserve the user's original meaning.
4. If the question is already standalone, return it
   unchanged.
5. Do not add information that is not present in the
   conversation.

CONVERSATION HISTORY:

{history}

CURRENT QUESTION:

{question}

STANDALONE QUESTION:
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
    # FORMAT CHAT HISTORY
    # --------------------------------------------------

    def format_history(messages):

        if not messages:
            return "No previous conversation."

        history_parts = []

        for message in messages:

            history_parts.append(
                f"User: {message['question']}"
            )

            history_parts.append(
                f"Assistant: {message['answer']}"
            )

        return "\n".join(history_parts)

    # --------------------------------------------------
    # ASK QUESTION
    # --------------------------------------------------

    def ask_question(
        question,
        conversation_history=None
    ):

        question_clean = question.strip()

        if conversation_history is None:
            conversation_history = []

        history = format_history(
            conversation_history
        )

        question_lower = question_clean.lower()

        # ---------------------------------------------
        # CHECK OVERVIEW QUESTION
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

            context = "\n\n".join(
                item["content"]
                for item in all_documents
                if item.get("content")
            )

            messages = overview_prompt.invoke(
                {
                    "context": context,
                    "question": question_clean
                }
            )

            # Only display a reasonable number
            # of source chunks.

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

            # -----------------------------------------
            # REWRITE QUESTION IF HISTORY EXISTS
            # -----------------------------------------

            search_question = question_clean

            if conversation_history:

                rewrite_messages = rewrite_prompt.invoke(
                    {
                        "history": history,
                        "question": question_clean
                    }
                )

                rewritten_response = llm.invoke(
                    rewrite_messages
                )

                search_question = extract_text(
                    rewritten_response
                )

            # -----------------------------------------
            # RETRIEVE DOCUMENTS
            # -----------------------------------------

            documents = retriever.invoke(
                search_question
            )

            if not documents:

                return {
                    "answer": (
                        "I don't know based on the "
                        "provided document."
                    ),
                    "sources": []
                }

            # -----------------------------------------
            # CREATE CONTEXT
            # -----------------------------------------

            context = "\n\n".join(
                document.page_content
                for document in documents
            )

            # -----------------------------------------
            # QA PROMPT
            # -----------------------------------------

            messages = qa_prompt.invoke(
                {
                    "history": history,
                    "context": context,
                    "question": question_clean
                }
            )

            # -----------------------------------------
            # SOURCES
            # -----------------------------------------

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
        # EXTRACT ANSWER
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
    # RETURN FUNCTION
    # --------------------------------------------------

    return ask_question