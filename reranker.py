from sentence_transformers import CrossEncoder
from functools import lru_cache

@lru_cache(maxsize=1)
def create_reranker():
    reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return reranker_model

def score_documents (reranker, user_query,candidate_docs):
    query_document_pairs = [(user_query,candidate_doc.page_content) for candidate_doc in candidate_docs ]

    # Initially I wanted to write the logic like this:
    # create an empty scored-documents list
    # for each candidate document:
    #   create one pair:
    #       user query + candidate document text
    #   send that pair to reranker.predict(...)
    #   receive its score
    #   add (candidate document, score) to scored-documents
    # return scored-documents
    #  But After considerations, batching is much more optimized than making an API call for each candidate.

    scores = reranker.predict(query_document_pairs)
    scored_docs = zip(candidate_docs, scores)

    return list(scored_docs)

def rank_scored_documents (scored_docs):

    return sorted(scored_docs,key=lambda item: item[1],reverse=True)