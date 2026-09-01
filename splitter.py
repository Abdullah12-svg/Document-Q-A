
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):

    # --------------------------------------------------
    # CSV
    # --------------------------------------------------
    # CSVLoader already creates one Document per row.
    # We should NOT split CSV rows further.

    if documents:

        source = documents[0].metadata.get(
            "source",
            ""
        )

        if source.lower().endswith(".csv"):

            return documents


    # --------------------------------------------------
    # PDF / TXT
    # --------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(
        documents
    )