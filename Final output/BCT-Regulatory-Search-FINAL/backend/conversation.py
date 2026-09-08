import json
from enum import Enum
from llm import create_llm
from answer_contract import generate_grounded_answer, safe_response, search_response, evidence_records
from vector_store import retrieve_relevant_chunks
from reranker import score_documents, rank_scored_documents
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from source_metadata import normalize_page
from retrieval_selection import parse_source_identity
from bm25 import retrieve_bm25
from pydantic import BaseModel, ConfigDict, model_validator
from graph_contract import (
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


def _normalize_route_payload(payload: dict, message: str) -> dict:
    """Repair common local-LLM type mistakes before schema validation."""
    data = dict(payload)
    intent = str(data.get("intent") or "").strip().upper()
    data["intent"] = intent

    for key in ("new_topic", "current_topic"):
        value = data.get(key)
        if isinstance(value, bool):
            data[key] = (str(data.get("rewrite_query") or message).strip()[:160] or None) if value else None
        elif value is not None and not isinstance(value, str):
            data[key] = str(value)

    if intent in {RouteIntent.NEW_TOPIC.value, RouteIntent.FOLLOW_UP.value}:
        rewrite = data.get("rewrite_query")
        if not isinstance(rewrite, str) or not rewrite.strip():
            data["rewrite_query"] = message
        if intent == RouteIntent.NEW_TOPIC.value and not data.get("new_topic"):
            data["new_topic"] = str(data.get("rewrite_query") or message).strip()[:160]
        if intent == RouteIntent.FOLLOW_UP.value and not data.get("current_topic"):
            data["current_topic"] = str(data.get("rewrite_query") or message).strip()[:160]
    return data





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
        - A request explaining how you can help, without a specific regulatory fact to look up, is GENERAL_CHAT.
        - Keep rewrite_query in the language of the current user message.
        - For NEW_TOPIC and FOLLOW_UP, rewrite_query must be a complete standalone search query with resolved document names, provisions, and dates from memory when available.
        - current_topic should be the topic the message refers to now.
        - new_topic and current_topic must be short topic strings, never booleans.
        - Example NEW_TOPIC JSON:
          {{"intent":"NEW_TOPIC","rewrite_query":"heures d'ouverture du marche des changes selon circulaire 2016-01","new_topic":"Horaires marche des changes 2016-01","current_topic":null}}
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
        payload = _normalize_route_payload(json.loads(result), message)
        route = MessageRoute.model_validate(payload)
    except (json.JSONDecodeError, ValueError):
        return _ambiguous_route()
    if route.intent == RouteIntent.FOLLOW_UP and not (
        memory_state.get("turns") or memory_state.get("current_topic")
    ):
        return _ambiguous_route()
    if (
        route.intent == RouteIntent.FOLLOW_UP
        and len(memory_state.get("topics", [])) > 1
        and not memory_state.get("current_topic")
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
    answer_status=None,
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
            "answer_status": answer_status,
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
    return safe_response(message, "clarification_needed")["answer"]

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


def _instrument_year(document):
    identity = parse_source_identity(str(document.metadata.get("source") or ""))
    return identity["year"] if identity else 0


def _prefer_later_instrument_evidence(results):
    return sorted(
        results,
        key=lambda item: (_instrument_year(item[0]), float(item[1])),
        reverse=True,
    )


def _answer_results(
    reranked_results,
    *,
    graph_results=(),
    ordinary_limit=5,
    graph_limit=2,
    prefer_later_instruments=False,
):
    """Assemble answer context without letting graph evidence displace normal RAG.

    The historical graph experiment showed that mixing graph snippets into the same
    five-slot reranking budget could recover a useful page and then lose it again.
    Graph Lite therefore reserves at most two verified graph-evidence slots while
    preserving the ordinary top five. Full provision-level VERIFIED temporal context,
    if a future resolver supplies it, remains mandatory.
    """
    ordinary = list(reranked_results[:ordinary_limit])
    if prefer_later_instruments:
        ordinary = _prefer_later_instrument_evidence(ordinary)
    graph = list(graph_results[:graph_limit])

    temporal = [
        item
        for item in ordinary + graph
        if item[0].metadata.get("temporal_resolution") == "VERIFIED"
    ]
    if temporal:
        mandatory = temporal[:1]
        mandatory_ids = {id(document) for document, _ in mandatory}
        remaining = [
            item
            for item in graph + ordinary
            if id(item[0]) not in mandatory_ids
        ]
        return mandatory + remaining[: ordinary_limit + graph_limit - len(mandatory)]

    # Put verified graph evidence first so the answer model sees the relationship
    # contract explicitly, while retaining the full ordinary retrieval budget.
    return graph + ordinary


def chat(
    message,
    memory_state,
    vector_store,
    reranker,
    bm25,
    bm25_documents,
    graph_retriever=None,
    retrieval_backend=None,
    llm_provider="groq",
):
    llm = create_llm() if llm_provider == "groq" else create_llm(llm_provider)

    route = route_message(llm, message, memory_state)

    if route["intent"] == "GENERAL_CHAT":
        refusal = safe_response(message, "out_of_scope")
        return {
            "answer": refusal["answer"],
            "sources": refusal["sources"],
            "status": refusal["status"],
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

    route_query = route["rewrite_query"] or message
    query_for_retrieval = (
        message
        if route["intent"] == RouteIntent.NEW_TOPIC.value
        else route_query
    )
    temporal_graph_query = (
        message
        if is_temporal_rule_query(message)
        else route_query
        if is_temporal_rule_query(route_query)
        else None
    )

    if retrieval_backend is None:
        retrieved_docs = retrieve_relevant_chunks(query_for_retrieval, vector_store)
        bm25_docs = retrieve_bm25(query_for_retrieval, bm25, bm25_documents)
        candidate_docs = combine_documents(retrieved_docs, bm25_docs)
        reranked_results = _rank_candidates(
            reranker,
            query_for_retrieval,
            candidate_docs,
        )
    else:
        reranked_results = retrieval_backend.retrieve(query_for_retrieval)
        candidate_docs = [document for document, _score in reranked_results]

    temporal_unverified = temporal_graph_query is not None and graph_retriever is None
    graph_trace = (
        GraphRetrievalTrace(
            status=GraphRetrievalStatus.UNAVAILABLE,
            temporal_status=TemporalRetrievalStatus.UNAVAILABLE,
            temporal_reason=TemporalFailureReason.TEMPORAL_GRAPH_UNAVAILABLE,
        ) if temporal_unverified else
        GraphRetrievalTrace(status=GraphRetrievalStatus.NOT_REQUESTED)
    )
    graph_ranked_results = []
    if graph_retriever is not None:
        seed_documents = [document for document, _ in reranked_results[:5]]
        graph_result = graph_retriever.retrieve(
            temporal_graph_query or route_query,
            seed_documents,
        )
        graph_trace = graph_result.trace
        if temporal_graph_query is not None:
            temporal_unverified = (
                graph_result.trace.temporal_status
                is not TemporalRetrievalStatus.RESOLVED
            )
        else:
            temporal_unverified = graph_result.requires_temporal_abstention
        # Keep verified relationship evidence in a separate, tiny budget. This
        # prevents ordinary page-diversity/reranking from discarding the exact
        # graph evidence that triggered the relationship result.
        if graph_result.documents:
            # Verified relationship quotes already passed the evidence gate.
            # Do not re-rank/diversify them: same-source edges share a page key and
            # ordinary diversify would collapse distinct AMENDS/CITES quotes to one.
            graph_ranked_results = [
                (document, float(len(graph_result.documents) - index))
                for index, document in enumerate(graph_result.documents)
            ]

    top_results = _answer_results(
        reranked_results,
        graph_results=graph_ranked_results,
        prefer_later_instruments=temporal_graph_query is not None,
    )
    memory_text = render_memory_state(_answer_memory(memory_state, route))
    if route["intent"] == RouteIntent.FOLLOW_UP.value:
        memory_text = f"Resolved reference (not factual evidence): {query_for_retrieval}\n\n{memory_text}"
    generated = generate_grounded_answer(
        llm,
        message,
        top_results,
        memory_text,
        **({"temporal_unverified": temporal_unverified} if temporal_graph_query is not None or temporal_unverified else {}),
    )
    if generated.get("status") in {"search_results", "insufficient_evidence", "clarification_needed"}:
        # Search fallback preserves ordinary retrieval order, even when answer
        # context was reordered for currentness or prefixed with graph snippets.
        generated = search_response(message, evidence_records(reranked_results[:5]))
    answer = generated["answer"]
    sources = generated["sources"]
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
            answer_status=generated.get("status"),
        ),
        "graph_trace": trace,
        "status": generated.get("status", "answered"),
    }
