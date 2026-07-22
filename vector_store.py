from langchain_core.vectorstores import InMemoryVectorStore

def create_vector_store(documents,embedding_model):
    vector_store = InMemoryVectorStore(
        embedding = embedding_model
    )

    vector_store.add_documents(documents)

    return vector_store

# Searches for the four most similar document chunks
# and includes each chunk's similarity score.
def retrieve_relevant_chunks(user_query,vector_store):
    results = vector_store.similarity_search_with_score(user_query, k=4)
    #Finds the document chunks that has a relevant score of at least 0.55
    best_docs = [document for document, score in results if score >= 0.55]

    return best_docs
