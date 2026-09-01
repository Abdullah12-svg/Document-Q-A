from loaders import load_document
from splitter import split_documents
from vectorstore import create_vectorstore


# Load CSV
documents = load_document("data/employee.csv")

print("\n========== LOADED DOCUMENTS ==========")
print("Number of documents:", len(documents))

for i, doc in enumerate(documents):
    print(f"\n--- Document {i + 1} ---")
    print(doc.page_content)


# Split
chunks = split_documents(documents)

print("\n========== CHUNKS ==========")
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)


# Create vectorstore
vectorstore = create_vectorstore(
    chunks,
    "debug_employees"
)


# Search
results = vectorstore.similarity_search(
    "What is this document about?",
    k=20
)


print("\n========== RETRIEVED RESULTS ==========")
print("Number of results:", len(results))

for i, result in enumerate(results):

    print(f"\n--- Retrieved {i + 1} ---")

    print(result.page_content)

    print("Metadata:")
    print(result.metadata)