import json
from enum import Enum
from llm import create_llm, generate_answer
from vector_store import retrieve_relevant_chunks
from reranker import score_documents, rank_scored_documents
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from bm25 import retrieve_bm25
from pydantic import BaseModel, ConfigDict, model_validator
from regulatory_graph.runtime import (
    GraphRetrievalStatus,
    GraphRetrievalTrace,
    TemporalFailureReason,
    TemporalRetrievalStatus,
    is_temporal_rule_query,
)


class RouteIntent(str, Enum):
    GENERAL_CHAT = "GENERAL_CHAT"
    NEW_TOPIC = "NEW_TOPIC"
    FOLLOW_UP = "FOLLOW_UP"
    AMBIGUOUS = "AMBIGUOUS"


class MessageRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: RouteIntent
    rewrite_query: str | None = None
    new_topic: str | None = None
    current_topic: str | None = None

    @model_validator(mode="after")
    def validate_retrieval_route(self):
        if self.intent in {RouteIntent.NEW_TOPIC, RouteIntent.FOLLOW_UP}:
            if not self.rewrite_query or not self.rewrite_query.strip():
                raise ValueError("retrieval route requires a standalone rewrite_query")
        if self.intent == RouteIntent.NEW_TOPIC and not self.new_topic:
            raise ValueError("NEW_TOPIC requires new_topic")
        if self.intent == RouteIntent.FOLLOW_UP and not self.current_topic:
            raise ValueError("FOLLOW_UP requires current_topic")
        return self


def _ambiguous_route():
    return MessageRoute(intent=RouteIntent.AMBIGUOUS).model_dump(mode="json")



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

    summary = (
        f"First topic: {first_topic}\n"
        f"Current topic: {current_topic}\n"
        f"Topics discussed: {', '.join(topics) if topics else 'None'}"
    )
    turns = memory_state.get("turns", [])[-4:]
    if not turns:
        return summary
    rendered_turns = []
    for turn in turns:
        sources = ", ".join(
            f"{source.get('file')} p.{source.get('page')}"
            for source in turn.get("sources", [])
        ) or "None"
        rendered_turns.append(
            "User: {user}\nStandalone query: {query}\nAnswer: {answer}\nSources: {sources}".format(
                user=turn.get("user_message", ""),
                query=turn.get("standalone_query", ""),
                answer=turn.get("answer", ""),
                sources=sources,
            )
        )
    return f"{summary}\n\nRecent turns:\n" + "\n\n".join(rendered_turns)


def route_message(llm, message, memory_state):

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a routing assistant for a regulatory search chatbot.

        Your job is to decide whether the message is:
        - GENERAL_CHAT
        - NEW_TOPIC
        - FOLLOW_UP
        - AMBIGUOUS

        Return only valid JSON with these keys:
        intent
        rewrite_query
        new_topic
        current_topic

        Rules:
        - If the user refers to the first circular, previous circular, that circular, these ones, etc., use FOLLOW_UP.
        - If the user changes to a different circular/topic, use NEW_TOPIC.
        - If a reference could point to more than one discussed topic and cannot be resolved safely, use AMBIGUOUS.
        - If the message is a greeting or identity question, use GENERAL_CHAT.
        - For NEW_TOPIC and FOLLOW_UP, rewrite_query must be a complete standalone search query with resolved document names, provisions, and dates from memory when available.
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

    result = result.strip()

    if result.startswith("```"):
        result = result.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        route = MessageRoute.model_validate(json.loads(result))
    except (json.JSONDecodeError, ValueError):
        return _ambiguous_route()
    if route.intent == RouteIntent.FOLLOW_UP and not (
        memory_state.get("turns") or memory_state.get("current_topic")
    ):
        return _ambiguous_route()
    return route.model_dump(mode="json")


def update_memory_state(
    memory_state,
    route,
    *,
    message=None,
    standalone_query=None,
    answer=None,
    sources=None,
    graph_trace=None,
):
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

    turns = list(memory_state.get("turns", []))
    if message is not None:
        turns.append({
            "user_message": message,
            "standalone_query": standalone_query or message,
            "answer": answer or "",
            "sources": list(sources or []),
            "graph_trace": dict(graph_trace or {}),
        })

    return {
        "topics": topics,
        "first_topic": first_topic,
        "current_topic": current_topic,
        "turns": turns[-6:],
    }


def _answer_memory(memory_state, route):
    if route.get("intent") != RouteIntent.NEW_TOPIC.value:
        return memory_state
    topic = route.get("new_topic") or route.get("current_topic")
    return {
        "topics": [topic] if topic else [],
        "first_topic": topic,
        "current_topic": topic,
        "turns": [],
    }


def _clarification_message(message):
    if any("\u0600" <= character <= "\u06ff" for character in message):
        return "يرجى تحديد المنشور أو المذكرة أو الموضوع الذي تشير إليه."
    normalized = message.casefold()
    if any(marker in normalized for marker in ("quel", "quelle", "circulaire", "précédent")):
        return "Veuillez préciser à quelle circulaire, note ou disposition vous faites référence."
    return "Please specify which circular, note, or provision you mean."

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


def _rank_candidates(reranker, query, documents):
    scored_documents = score_documents(reranker, query, documents)
    return rank_scored_documents(scored_documents)


def _temporal_abstention_message(message):
    if any("\u0600" <= character <= "\u06ff" for character in message):
        return (
            "لا يمكنني تحديد الحكم النافذ بأمان لأن التسلسل الزمني الموثق "
            "في الرسم البياني غير متاح أو غير مكتمل."
        )
    normalized = message.casefold()
    if any(
        marker in normalized
        for marker in ("règle", "circulaire", "en vigueur", "disposition")
    ):
        return (
            "Je ne peux pas déterminer la disposition applicable de manière "
            "fiable, car la chronologie vérifiée du graphe est indisponible "
            "ou incomplète."
        )
    return (
        "I cannot determine the controlling provision safely because the "
        "verified temporal graph lineage is unavailable or incomplete."
    )


def _answer_results(reranked_results, *, limit=5):
    temporal = [
        item
        for item in reranked_results
        if item[0].metadata.get("temporal_resolution") == "VERIFIED"
    ]
    if not temporal:
        return reranked_results[:limit]
    mandatory = temporal[:1]
    mandatory_ids = {id(document) for document, _ in mandatory}
    remainder = [
        item
        for item in reranked_results
        if id(item[0]) not in mandatory_ids
    ]
    return mandatory + remainder[: limit - len(mandatory)]


def chat(
    message,
    memory_state,
    vector_store,
    reranker,
    bm25,
    bm25_documents,
    graph_retriever=None,
):
    llm = create_llm()

    route = route_message(llm, message, memory_state)

    if route["intent"] == "GENERAL_CHAT":
        return {
            "answer": "Hi! I'm here to help with Banque Centrale documents.",
            "sources": [],
            "memory_state": memory_state,
            "graph_trace": GraphRetrievalTrace(
                status=GraphRetrievalStatus.NOT_REQUESTED
            ).as_dict(),
        }

    if route["intent"] == "AMBIGUOUS":
        return {
            "answer": _clarification_message(message),
            "sources": [],
            "memory_state": memory_state,
            "graph_trace": GraphRetrievalTrace(
                status=GraphRetrievalStatus.NOT_REQUESTED
            ).as_dict(),
        }

    query_for_retrieval = route["rewrite_query"] or message
    temporal_graph_query = (
        message
        if is_temporal_rule_query(message)
        else query_for_retrieval
        if is_temporal_rule_query(query_for_retrieval)
        else None
    )

    if temporal_graph_query is not None and graph_retriever is None:
        graph_trace = GraphRetrievalTrace(
            status=GraphRetrievalStatus.UNAVAILABLE,
            temporal_status=TemporalRetrievalStatus.UNAVAILABLE,
            temporal_reason=TemporalFailureReason.TEMPORAL_GRAPH_UNAVAILABLE,
        )
        answer = _temporal_abstention_message(message)
        trace = graph_trace.as_dict()
        return {
            "answer": answer,
            "sources": [],
            "memory_state": update_memory_state(
                memory_state,
                route,
                message=message,
                standalone_query=query_for_retrieval,
                answer=answer,
                sources=[],
                graph_trace=trace,
            ),
            "graph_trace": trace,
        }

    retrieved_docs = retrieve_relevant_chunks(query_for_retrieval, vector_store)

    bm25_docs = retrieve_bm25(query_for_retrieval, bm25, bm25_documents)

    candidate_docs = combine_documents(retrieved_docs, bm25_docs)
    reranked_results = _rank_candidates(
        reranker,
        query_for_retrieval,
        candidate_docs,
    )

    graph_trace = GraphRetrievalTrace(status=GraphRetrievalStatus.NOT_REQUESTED)
    if graph_retriever is not None:
        seed_documents = [document for document, _ in reranked_results[:5]]
        graph_result = graph_retriever.retrieve(
            temporal_graph_query or query_for_retrieval,
            seed_documents,
        )
        graph_trace = graph_result.trace
        if graph_result.requires_temporal_abstention:
            answer = _temporal_abstention_message(message)
            trace = graph_trace.as_dict()
            return {
                "answer": answer,
                "sources": [],
                "memory_state": update_memory_state(
                    memory_state,
                    route,
                    message=message,
                    standalone_query=query_for_retrieval,
                    answer=answer,
                    sources=[],
                    graph_trace=trace,
                ),
                "graph_trace": trace,
            }
        if graph_result.documents:
            candidate_docs = combine_documents(
                candidate_docs,
                list(graph_result.documents),
            )
            reranked_results = _rank_candidates(
                reranker,
                query_for_retrieval,
                candidate_docs,
            )

    top_results = _answer_results(reranked_results)
    documents_for_llm = [document for document, _ in top_results]

    memory_text = render_memory_state(_answer_memory(memory_state, route))
    answer = generate_answer(
        llm,
        query_for_retrieval,
        documents_for_llm,
        memory_text,
    )
    sources = build_sources(top_results)
    trace = graph_trace.as_dict()

    return {
        "answer": answer,
        "sources": sources,
        "memory_state": update_memory_state(
            memory_state,
            route,
            message=message,
            standalone_query=query_for_retrieval,
            answer=answer,
            sources=sources,
            graph_trace=trace,
        ),
        "graph_trace": trace,
    }
