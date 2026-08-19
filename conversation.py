import json
from llm import create_llm, generate_answer
from vector_store import retrieve_relevant_chunks
from reranker import score_documents, rank_scored_documents
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from bm25 import retrieve_bm25



def build_sources(scored_documents):
    sources = []
    seen = set()

    for document, score in scored_documents:
        metadata = document.metadata
        filename = Path(metadata.get("source", "")).name
        page = normalize_page(metadata)

        if (filename, page) in seen:
            continue
        seen.add((filename,page))

        sources.append({
            "file": filename,
            "page": page,
            "score": round(float(score), 4),
        })

    return sources

def normalize_page(metadata):
    label = metadata.get("page_label")
    if label is not None:
        try:
            return int(label)
        except (TypeError,ValueError):
            return label

    page = metadata.get("page")
    return page + 1 if isinstance(page, int) else page

def is_general_conversation(message):
    return {} #post poned


def render_memory_state(memory_state):
    topics = memory_state.get("topics", [])
    first_topic = memory_state.get("first_topic") or "None"
    current_topic = memory_state.get("current_topic") or "None"

    return (
        f"First topic: {first_topic}\n"
        f"Current topic: {current_topic}\n"
        f"Topics discussed: {', '.join(topics) if topics else 'None'}"
    )


def route_message(llm, message, memory_state):

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a routing assistant for a regulatory search chatbot.

        Your job is to decide whether the message is:
        - GENERAL_CHAT
        - NEW_TOPIC
        - FOLLOW_UP

        Return only valid JSON with these keys:
        intent
        rewrite_query
        new_topic
        current_topic

        Rules:
        - If the user refers to the first circular, previous circular, that circular, these ones, etc., use FOLLOW_UP.
        - If the user changes to a different circular/topic, use NEW_TOPIC.
        - If the message is a greeting or identity question, use GENERAL_CHAT.
        - rewrite_query should be the cleaned search query to use for retrieval.
        - current_topic should be the topic the message refers to now.
                """),
        ("human",
         "Current memory:\n{memory_text}\n\n"
         "User message:\n{message}")
    ])

    
    chain = prompt | llm
    result = chain.invoke({
        "memory_text": render_memory_state(memory_state),
        "message": message,
    }).content

    print("ROUTER RAW OUTPUT:", repr(result))

    result = result.strip()

    if result.startswith("```"):
        result = result.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(result)


def update_memory_state(memory_state, route):
    topics = list(memory_state.get("topics", []))
    first_topic = memory_state.get("first_topic")
    current_topic = memory_state.get("current_topic")

    if route.get("intent") == "NEW_TOPIC" and route.get("new_topic"):
        current_topic = route["new_topic"]
        if current_topic not in topics:
            topics.append(current_topic)
        if first_topic is None:
            first_topic = current_topic

    elif route.get("intent") == "FOLLOW_UP" and route.get("current_topic"):
        current_topic = route["current_topic"]
        if current_topic not in topics:
            topics.append(current_topic)
        if first_topic is None and topics:
            first_topic = topics[0]

    return {
        "topics": topics,
        "first_topic": first_topic,
        "current_topic": current_topic,
    }

# Combine both the chroma documents and the bm25 documents into one data structure and avoid repetition. 
def combine_documents(chroma_docs, bm25_docs):
    combined_docs = []
    seen = set()

    for document in chroma_docs + bm25_docs:

        key = (
            document.page_content,
            document.metadata.get("source"),
            document.metadata.get("page")
        )

        if key not in seen:
            seen.add(key)
            combined_docs.append(document)

    return combined_docs
def chat(message, memory_state, vector_store, reranker, bm25, bm25_documents):
    llm = create_llm()

    route = route_message(llm, message, memory_state)

    if route["intent"] == "GENERAL_CHAT":
        return {
            "answer": "Hi! I'm here to help with Banque Centrale documents.",
            "sources": [],
            "memory_state": memory_state,
        }

    query_for_retrieval = route["rewrite_query"] or message

    retrieved_docs = retrieve_relevant_chunks(query_for_retrieval, vector_store)

    bm25_docs = retrieve_bm25(query_for_retrieval, bm25, bm25_documents)

    candidate_docs = combine_documents(retrieved_docs, bm25_docs)


    
    scored_docs = score_documents(reranker, query_for_retrieval, candidate_docs)
    reranked_results = rank_scored_documents(scored_docs)

    top_results = reranked_results[:5]
    documents_for_llm = [document for document, _ in top_results]

    memory_text = render_memory_state(memory_state)
    answer = generate_answer(llm, message, documents_for_llm, memory_text)

    return {
        "answer": answer,
        "sources": build_sources(top_results),
        "memory_state": update_memory_state(memory_state, route),
    }
