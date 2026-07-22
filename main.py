from load_pdf import load_pdf
from chunk_pdf import chunk_pdf
from embedding import create_embedding_model
from vector_store import ( create_vector_store, retrieve_relevant_chunks )
from llm import create_llm,generate_answer

pages = load_pdf("documents/Circulaires et notes 2026/Cir_2026_01_fr.pdf")
chunks = chunk_pdf(pages)

query = "Quel est le prix d'or ?"
 
embedding_model = create_embedding_model()

vector_store = create_vector_store(chunks,embedding_model)

result = retrieve_relevant_chunks(query,vector_store)

if not result:
    print ("No relevant document found.")
else:
    print (result[0].page_content)

llm = create_llm()
response = generate_answer(llm,query,result)
print (response.content)