from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

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

def create_bm25(documents):

    tokenized_documents = tokenize_documents(documents)

    bm25 = BM25Okapi(tokenized_documents)

    return bm25

def retrieve_bm25(query, bm25, documents, k=15):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [documents[i] for i in top_indices]