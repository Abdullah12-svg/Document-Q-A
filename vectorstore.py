
from langchain_chroma import Chroma

from embeddings import get_embeddings


# ==================================================
# CREATE VECTOR STORE
# ==================================================

def create_vectorstore(documents, collection_name):

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name=collection_name
    )

    return vectorstore


# ==================================================
# CREATE RETRIEVER
# ==================================================

def create_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
            "lambda_mult": 0.7
        }
    )

# ==================================================
# SEARCH WITH RELEVANCE SCORES
# ==================================================

def search_with_scores(
    vectorstore,
    query,
    k=5,
    score_threshold=0.7
):

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=k
    )

    filtered_documents = []

    for document, score in results:

        if score >= score_threshold:

            filtered_documents.append(
                document
            )

    return filtered_documents



# ==================================================
# GET ALL DOCUMENT CHUNKS
# ==================================================

def get_all_documents(vectorstore):

    collection_data = vectorstore.get()

    documents = []

    documents_data = collection_data.get(
        "documents",
        []
    )

    metadatas_data = collection_data.get(
        "metadatas",
        []
    )

    for index, content in enumerate(
        documents_data
    ):

        metadata = {}

        if index < len(metadatas_data):

            metadata = (
                metadatas_data[index]
                or {}
            )

        documents.append(
            {
                "content": content,
                "metadata": metadata
            }
        )

    return documents
