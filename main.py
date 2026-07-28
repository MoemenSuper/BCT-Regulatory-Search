from vector_store import retrieve_relevant_chunks 
from llm import (create_llm, generate_answer)
from reranker import (create_reranker, score_documents, rank_scored_documents)
from ingestion import ingest

vector_store = ingest()
query = "Quels sont les horaires habituels du marché interbancaire des changes ?"


result = retrieve_relevant_chunks(query,vector_store)

if not result:
    print ("No relevant document found.")

else:
    print (result[0].page_content)
    reranker = create_reranker()
    scored_docs = score_documents (reranker,query,result)
    reranked_results = rank_scored_documents(scored_docs)
    documents_for_llm = [
        document
        for document, _ in reranked_results
    ]
    print ("\n\n##############################################################\n\n")
    llm = create_llm()
    response = generate_answer(llm,query,documents_for_llm)
    print (response.content)


