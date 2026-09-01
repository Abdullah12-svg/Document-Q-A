from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
)


def load_document(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".csv":
        loader = CSVLoader(str(path))

    elif extension == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8"
        )

    else:
        raise ValueError(
            "Unsupported file type. "
            "Only PDF, CSV, and TXT files are supported."
        )

    return loader.load()