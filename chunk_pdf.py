from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("documents/CONDITIONS DE BANQUE.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

print(f"Number of pages: {len(pages)}")
print(f"Number of chunks: {len(chunks)}")

print("\nFirst chunk:")
print(chunks[0].page_content)

print("\nFirst chunk metadata:")
print(chunks[0].metadata)