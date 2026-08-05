from llm import create_llm,generate_answer
from vector_store import retrieve_relevant_chunks
from reranker import create_reranker,score_documents,rank_scored_documents

def chat(message, history, memory_summary, vector_store):
    # 1) rebuild context from history
    recent_context = get_recent_context(history,4)
    query_for_retrieval = message
    if recent_context.strip():
        query_for_retrieval = recent_context + "\nCurrent question: " + message
    # 2) retrieve docs for the current message
    retrieved_docs = retrieve_relevant_chunks(query_for_retrieval, vector_store)

    # 3) rerank docs
    reranker = create_reranker()
    scored_docs = score_documents(reranker, query_for_retrieval, retrieved_docs)
    reranked_results = rank_scored_documents(scored_docs)
    documents_for_llm = [
        document
        for document, _ in reranked_results
    ]

    # 4) generate answer
    llm = create_llm()
    response = generate_answer(llm, message, documents_for_llm)

    # 5) update memory_summary
    memory_summary = update_memory_summary(memory_summary, message, response)
    # 6) return answer + memory_summary
    return response, memory_summary


def get_recent_context(history, n=4):
    recent = history[-n:]
    text = ""

    for item in recent:
        # format 1: tuple/list like ("user", "assistant")
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            text += f"User: {user_msg}\nAssistant: {bot_msg}\n"

        # format 2: dict like {"role": "...", "content": "..."}
        elif isinstance(item, dict):
            role = item.get("role", "")
            content = item.get("content", "")
            text += f"{role}: {content}\n"

        # fallback
        else:
            text += f"{str(item)}\n"

    return text

def update_memory_summary(memory_summary, message, response):
    return f"Last user topic: {message}\nLast answer: {response[:200]}"