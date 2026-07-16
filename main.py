from load_pdf import load_pdf
from chunk_pdf import chunk_pdf
from embedding import create_embedding_model


pages = load_pdf("documents/Circulaires et notes 2026/Cir_2026_01_fr.pdf")
chunks = chunk_pdf(pages)

chunk_texts = [chunk.page_content for chunk in chunks]
 
embedding_model = create_embedding_model()

embeddings = embedding_model.embed_documents(chunk_texts)

print("\n\n ########################################################### \n\n")
print(embeddings)
