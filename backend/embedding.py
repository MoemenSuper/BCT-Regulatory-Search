from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        encode_kwargs={"prompt": "passage: "},
        query_encode_kwargs={"prompt": "query: "},
    )
