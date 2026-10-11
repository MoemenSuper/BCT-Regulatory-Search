"""The embedding model: turns a text into a vector (a list of numbers) so that texts with close
meanings get close vectors. Used twice: once per chunk at ingestion (stored in Chroma), and once
per question at search time (step 3 of a question: "meaning search").

The model is multilingual-e5-small, a bi-encoder: question and chunk are encoded separately,
which is what lets every chunk be encoded once, in advance.
"""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from rag.hardware import batch_size, torch_device


@lru_cache(maxsize=1)  # load the model once per process; every later call reuses it
def create_embedding_model():
    # e5 was trained with a prefix that says which side a text is on: "passage: " for the chunks
    # we store, "query: " for the question. Without the prefixes its vectors are worse.
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small",
        model_kwargs={"device": torch_device()},
        encode_kwargs={"prompt": "passage: ", "batch_size": batch_size(128)},
        query_encode_kwargs={"prompt": "query: "},
    )
