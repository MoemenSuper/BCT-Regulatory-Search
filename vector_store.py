from langchain_core.vectorstores import InMemoryVectorStore

def create_vector_store(documents,embedding_model):
    vector_store = InMemoryVectorStore(
        embedding = embedding_model
    )

    vector_store.add_documents(documents)

    return vector_store

def retrieve_relevant_chunks(user_query,vector_store):
    results = vector_store.similarity_search_with_score(user_query, k=4)

    best_doc,best_score = results[0]
    if (best_score < 0.55):
        return None

    return best_doc