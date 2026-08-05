from embedding import create_embedding_model
from vector_store import (retrieve_relevant_chunks,load_vector_store) 


embedding_model = create_embedding_model()
vector_store = load_vector_store(embedding_model)
query = "Que prévoit la circulaire 2026-2 pour les bureaux de change ?"


result = retrieve_relevant_chunks(query,vector_store)

if not result:
    print ("No relevant document found.")

else:
    print (result[0].page_content)
    print ("\n\n##############################################################\n\n")


