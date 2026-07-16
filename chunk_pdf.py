from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pdf(documents):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ) #Creates a text_splitter object using the RecursiveCharacterTextSplitter class. chunk size divides by CHARECTERS and 
    # chunk overlap represents how much charecters will be repeated from the previous chunk onto the next one.

    return text_splitter.split_documents(documents)
