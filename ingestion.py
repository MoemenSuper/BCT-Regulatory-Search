from pathlib import Path
from load_pdf import load_pdf
from chunk_pdf import chunk_pdf
from embedding import create_embedding_model
from vector_store import  create_vector_store

def ingest():
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
        return vector_store

if __name__ == "__main__":
    ingest()