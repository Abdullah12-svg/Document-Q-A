from loaders import load_document


documents = load_document("data/employee.csv")

print(f"Number of documents: {len(documents)}")

for i, document in enumerate(documents):

    print(f"\n--- Row {i + 1} ---")

    print("Content:")
    print(document.page_content)

    print("Metadata:")
    print(document.metadata)