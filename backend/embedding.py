from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from hardware import batch_size, torch_device


@lru_cache(maxsize=1)
def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": torch_device()},
        encode_kwargs={"prompt": "passage: ", "batch_size": batch_size(128)},
        query_encode_kwargs={"prompt": "query: "},
    )
