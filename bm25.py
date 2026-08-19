from langchain_core.documents import Document

def load_documents_from_chroma(vector_store):

    results = vector_store.get(include=["documents", "metadatas"])
    documents = []

    for text, metadata in zip(results["documents"], results["metadatas"]):

        document = Document(
            page_content = text,
            metadata = metadata
        )

        documents.append(document)

    return documents

def tokenize_documents(documents):
    tokenized_documents = []

    for document in documents:
        text = document.page_content.lower()
        tokens = text.split()

        tokenized_documents.append(tokens)

    return tokenized_documents