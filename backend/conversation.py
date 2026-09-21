import json
from enum import Enum
from llm import create_llm
from answer_contract import (
    format_refusal_reason,
    generate_grounded_answer,
    safe_response,
    search_response,
    evidence_records,
)
from langchain_core.prompts import ChatPromptTemplate
from retrieval_selection import (
    explicit_instrument_identity,
    parse_source_identity,
    prefer_named_instrument_hits,
    _instrument_year,
)
from pydantic import BaseModel, ConfigDict, model_validator
from graph_contract import (
    GraphRetrievalStatus,
    GraphRetrievalTrace,
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


_DEICTIC_MARKERS = (
    "cette opération",
    "cette operation",
    "cette activité",
    "cette activite",
    "pour ça",
    "pour ca",
    "pour cela",
    "cette-là",
    "celle-là",
    "celle-ci",
    "this operation",
    "for that",
    "هذه العملية",
    "هذه العملية؟",
)


def _looks_like_unresolved_deictic(message: str) -> bool:
    """Standalone messages that point at 'this/that' without naming the operation."""
    text = " ".join(str(message or "").casefold().split())
    if not text:
        return False
    return any(marker in text for marker in _DEICTIC_MARKERS)


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
        - Use GENERAL_CHAT for greetings, thanks, identity ("who are you"), how-you-work / what-can-you-do,
          requests to summarise or recall this conversation, and light off-topic chat that is not a BCT
          regulatory fact lookup. Do NOT send those to NEW_TOPIC.
        - A request explaining how you can help, without a specific regulatory fact to look up, is GENERAL_CHAT.
        - If the message refers to "this/that operation", "pour ça", "cette opération", or similar
          without a resolvable antecedent in memory, use AMBIGUOUS — never GENERAL_CHAT and never invent
          a topic. Do not greet the user as if they only said hello.
        - Keep rewrite_query in the language of the current user message.
        - For NEW_TOPIC and FOLLOW_UP, rewrite_query must be a complete standalone search query.
        - For NEW_TOPIC, rewrite only the current user message. Do not import facts, document names,
          provisions, dates, or topics from memory. For FOLLOW_UP, resolve references from memory when available.
        - For GENERAL_CHAT and AMBIGUOUS, rewrite_query, new_topic and current_topic must be null.
        - current_topic should be the topic the message refers to now.
        - new_topic and current_topic must be short topic strings, never booleans.
        - Example NEW_TOPIC JSON:
          {{"intent":"NEW_TOPIC","rewrite_query":"heures d'ouverture du marche des changes selon circulaire 2016-01","new_topic":"Horaires marche des changes 2016-01","current_topic":null}}
        - Example GENERAL_CHAT JSON:
          {{"intent":"GENERAL_CHAT","rewrite_query":null,"new_topic":null,"current_topic":null}}
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
    # Deictic regulatory asks with no prior turn must clarify, not greet.
    if (
        route.intent == RouteIntent.GENERAL_CHAT
        and _looks_like_unresolved_deictic(message)
        and not (memory_state.get("turns") or memory_state.get("current_topic"))
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


def _prefer_later_instrument_evidence(results):
    return sorted(
        results,
        key=lambda item: (_instrument_year(item[0]), float(item[1])),
        reverse=True,
    )


def _answer_results(
    reranked_results,
    *,
    ordinary_limit=5,
    prefer_later_instruments=False,
    query=None,
):
    ordinary = list(reranked_results[:ordinary_limit])
    if prefer_later_instruments:
        ordinary = _prefer_later_instrument_evidence(ordinary)
    identity = explicit_instrument_identity(query) if query else None
    if identity:
        ordinary = prefer_named_instrument_hits(ordinary, identity)
    return ordinary


def general_chat_reply(llm, message, memory_state):
    """No-retrieval assistant reply for greetings, help, and conversation summary."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the assistant for BCT Regulatory Search at the Banque Centrale de Tunisie
(Central Bank of Tunisia). Never invent another country or institution.
Answer in the same language as the user message. Be brief (a few sentences).
You may use conversation memory below. You have NO access to PDF text in this mode.
Allowed: greet the user; explain that you search Tunisian BCT circulars and notes with grounded
citations; summarise what was already discussed in this conversation from memory; politely decline
off-topic requests and invite a specific regulatory question.
Forbidden: invent circular numbers, pages, quotes, rates, or legal conclusions not present in memory.
Do not pretend you retrieved documents. Plain text only — no JSON, no markdown headings."""),
        ("human", "Conversation memory:\n{memory}\n\nUser message:\n{message}"),
    ])
    try:
        raw = (prompt | llm).invoke({
            "memory": render_memory_state(memory_state) or "(empty)",
            "message": message,
        }).content
    except Exception:
        return safe_response(message, "out_of_scope")
    answer = (raw or "").strip()
    if answer.startswith("```"):
        answer = answer.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if not answer:
        return safe_response(message, "out_of_scope")
    return {"status": "answered", "answer": answer, "sources": []}


def chat(
    message,
    memory_state,
    *,
    retrieval_backend,
    llm_provider="groq",
):
    llm = create_llm() if llm_provider == "groq" else create_llm(llm_provider)

    route = route_message(llm, message, memory_state)

    if route["intent"] in {"GENERAL_CHAT", "AMBIGUOUS"}:
        if route["intent"] == "GENERAL_CHAT":
            # ponytail: replace out_of_scope dead-end; no retrieval for assistant-style turns.
            reply = general_chat_reply(llm, message, memory_state)
            return {
                "answer": reply["answer"],
                "sources": reply["sources"],
                "status": reply["status"],
                "memory_state": memory_state,
                "graph_trace": GraphRetrievalTrace(
                    status=GraphRetrievalStatus.NOT_REQUESTED
                ).as_dict(),
                "refusal_reason": None,
                "refusal_diagnostics": [],
            }
        clarification = safe_response(message, "clarification_needed")
        return {
            "answer": clarification["answer"],
            "sources": [],
            "status": clarification["status"],
            "memory_state": memory_state,
            "graph_trace": GraphRetrievalTrace(
                status=GraphRetrievalStatus.NOT_REQUESTED
            ).as_dict(),
            "refusal_reason": format_refusal_reason(
                "clarification_needed",
                ["route:AMBIGUOUS:question_scope_unclear"],
            ),
            "refusal_diagnostics": ["route:AMBIGUOUS:question_scope_unclear"],
        }

    route_query = route["rewrite_query"] or message
    query_for_retrieval = route_query
    temporal_unverified = is_temporal_rule_query(message) or is_temporal_rule_query(
        route_query
    )

    reranked_results = retrieval_backend.retrieve(query_for_retrieval)
    # Opaque compatibility field for conversation memory / API clients.
    graph_trace = GraphRetrievalTrace(status=GraphRetrievalStatus.NOT_REQUESTED)

    # Retrieval is scored per chunk; the answer layer reads the whole retrieved page
    # so a fact in a neighbouring chunk is not lost. Order and citations are unchanged.
    expand_pages = getattr(retrieval_backend, "expand_pages", None)
    answer_candidates = reranked_results[:5]
    if expand_pages is not None:
        answer_candidates = expand_pages(answer_candidates)
    top_results = _answer_results(
        answer_candidates,
        prefer_later_instruments=temporal_unverified,
        query=message,
    )
    memory_text = render_memory_state(_answer_memory(memory_state, route))
    if route["intent"] == RouteIntent.FOLLOW_UP.value:
        memory_text = f"Resolved reference (not factual evidence): {query_for_retrieval}\n\n{memory_text}"
    generated = generate_grounded_answer(
        llm,
        message,
        top_results,
        memory_text,
        temporal_unverified=temporal_unverified,
    )
    diagnostics = list(generated.get("diagnostics") or [])
    status = generated.get("status", "answered")
    refusal_reason = None
    if status in {"search_results", "insufficient_evidence", "clarification_needed", "out_of_scope"}:
        refusal_reason = format_refusal_reason(status, diagnostics)
        if status in {"search_results", "insufficient_evidence"}:
            # Search fallback preserves ordinary retrieval order, even when answer
            # context was reordered for currentness.
            generated = search_response(message, evidence_records(reranked_results[:5]))
    answer = generated["answer"]
    sources = generated["sources"]
    trace = graph_trace.as_dict()

    result = {
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
    if refusal_reason:
        result["refusal_reason"] = refusal_reason
        result["refusal_diagnostics"] = diagnostics
    return result
