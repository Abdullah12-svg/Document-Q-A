from dotenv import load_dotenv

load_dotenv()

from loaders import load_document
from splitter import split_documents
from vectorstore import create_vectorstore
from qa import create_qa_chain


documents = load_document("data/sample.txt")

chunks = split_documents(documents)

vectorstore = create_vectorstore(chunks)

ask_question = create_qa_chain(vectorstore)


question = "What is Retrieval-Augmented Generation?"

result = ask_question(question)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(source)