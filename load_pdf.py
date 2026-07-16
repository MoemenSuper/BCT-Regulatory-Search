from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):

    loader = PyPDFLoader(file_path)
    return loader.load()

