from loaders import load_document
from splitter import split_documents
from vectorstore import create_vectorstore


documents = load_document("data/sample.txt")

chunks = split_documents(documents)

vectorstore = create_vectorstore(chunks)

print("Vector store created successfully!")

question = "What is Retrieval-Augmented Generation?"

results = vectorstore.similarity_search(
    question,
    k=2
)

print("\nSearch Results:")

for i, result in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)
    print("Metadata:", result.metadata)