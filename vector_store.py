from langchain_core.vectorstores import InMemoryVectorStore

def create_vector_store(documents,embedding_model):
    vector_store = InMemoryVectorStore(
        embedding = embedding_model
    )
    
    vector_store.add_documents(documents)

    return vector_store