from langchain_chroma import Chroma

def create_vector_store(documents,embedding_model):
    vector_store = InMemoryVectorStore(
        embedding = embedding_model
    )

    vector_store.add_documents(documents)

    return vector_store

# Searches for the 20 most similar document chunks
# and includes each chunk's similarity score.
def retrieve_relevant_chunks(user_query,vector_store):
    results = vector_store.similarity_search_with_score(user_query, k=20)
    return [
        document
        for document, _ in results
    ]
