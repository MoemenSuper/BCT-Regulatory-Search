from langchain_huggingface import HuggingFaceEmbeddings

class E5Embeddings(HuggingFaceEmbeddings):


    def embed_documents(self,texts):
        prefixed_texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(prefixed_texts)

    def embed_query(self,text):
        prefixed_text = f"query: {text}"
        return super().embed_query(prefixed_text)


def create_embedding_model():
    return E5Embeddings(
        model_name="intfloat/multilingual-e5-small"
    )