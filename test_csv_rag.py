from dotenv import load_dotenv

load_dotenv()

from loaders import load_document
from splitter import split_documents
from vectorstore import create_vectorstore
from qa import create_qa_chain


documents = load_document("data/employee.csv")

for document in documents:
    document.metadata["source"] = "employees.csv"


chunks = split_documents(documents)

vectorstore = create_vectorstore(
    chunks,
    "employees_csv"
)

ask_question = create_qa_chain(vectorstore)


questions = [
    "Who is the AI Engineer?",
    "Who has the highest salary?",
    "Which department does Umer Tariq work in?"
]


for question in questions:

    print("\n" + "=" * 50)

    print("Question:")
    print(question)

    result = ask_question(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    for source in result["sources"]:
        print(source)