from load_pdf import load_pdf
from chunk_pdf import chunk_pdf
from embedding import create_embedding_model
from vector_store import ( create_vector_store, retrieve_relevant_chunks )
from llm import (create_llm, generate_answer)
from pathlib import Path
from reranker import (create_reranker, score_documents, rank_scored_documents)

# For now we scan 399 pdfs which is 1554 pages which is 3290 chunks in total
documents_dir = Path("documents")
if (documents_dir.exists() and documents_dir.is_dir()):
     candidates = documents_dir.rglob("*")
     pages = []
     for candidate in candidates:
         if (candidate.is_file() and candidate.suffix == ".pdf"):
             try:
                 pages.extend(load_pdf(str(candidate)))

             except:
                 print("\t\t###Error loading PDF:\t\t\n ", candidate.stem)
    
     chunks = chunk_pdf(pages)
     embedding_model = create_embedding_model()
     vector_store = create_vector_store(chunks,embedding_model)


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


