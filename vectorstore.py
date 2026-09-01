
from langchain_chroma import Chroma

from embeddings import get_embeddings


# --------------------------------------------------
# CREATE VECTOR STORE
# --------------------------------------------------

def create_vectorstore(documents, collection_name):

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name=collection_name
    )

    return vectorstore


# --------------------------------------------------
# CREATE RETRIEVER
# --------------------------------------------------

def create_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )


# --------------------------------------------------
# GET ALL DOCUMENT CHUNKS
# --------------------------------------------------

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

