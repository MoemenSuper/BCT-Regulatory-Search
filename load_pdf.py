from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documents/CONDITIONS DE BANQUE.pdf")
pages = loader.load()

print(f"Number of pages: {len(pages)}  \n")

print(f"Metadata: ",pages[0].metadata," \n")

print(f"First Page content: ",pages[0].page_content)