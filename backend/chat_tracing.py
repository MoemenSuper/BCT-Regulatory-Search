"""Temporary Langfuse tracing for chat diagnosis. Delete this module and its call sites to remove."""
from contextlib import contextmanager
from pathlib import Path

from langchain_core.runnables import RunnableLambda
from langfuse import get_client, propagate_attributes


@contextmanager
def chat_turn(question, *, conversation_id, user_id, profile):
    langfuse = get_client()
    with langfuse.start_as_current_observation(name="chat-turn", as_type="chain", input=question) as root, propagate_attributes(
        session_id=f"chat-{conversation_id}", user_id=str(user_id), trace_name="chat-turn", tags=["chat", profile],
    ):
        yield root


def span(name, *, as_type="span", **observation):
    return get_client().start_as_current_observation(name=name, as_type=as_type, **observation)


def event(name, **observation):
    get_client().create_event(name=name, **observation)


def brief(records, limit=10):
    """Compact view of Documents or evidence records: source, page, score, snippet."""
    rows = []
    for record in list(records or [])[:limit]:
        if isinstance(record, tuple):
            record, score = record[0], record[1]
        else:
            score = None
        meta = getattr(record, "metadata", None)
        if meta is not None:
            text, source, page = record.page_content, meta.get("source"), meta.get("page")
            score = meta.get("rerank_score", score)
        else:
            text, source, page = record.get("text"), record.get("source"), record.get("page")
            score = record.get("score", score)
            if record.get("evidence_id"):
                source = f"{record['evidence_id']} {source}"
            if record.get("unusable_reason"):
                source = f"{source} [unusable:{record['unusable_reason']}]"
        rows.append({
            "source": Path(str(source or "")).name if "/" in str(source) or "\\" in str(source) else source,
            "page": page,
            "score": None if score is None else round(float(score), 4),
            "text": " ".join(str(text or "").split())[:240],
        })
    return rows


def traced_llm(llm):
    def call(value, config):
        messages = value.to_messages() if hasattr(value, "to_messages") else value
        prompt = (
            [{"role": getattr(m, "type", "user"), "content": str(m.content)} for m in messages]
            if isinstance(messages, list) else str(messages)
        )
        with span("llm", as_type="generation", input=prompt) as generation:
            try:
                result = llm.invoke(value, config=config)
            except Exception as error:
                generation.update(level="ERROR", status_message=f"{type(error).__name__}: {str(error)[:500]}")
                raise
            meta = getattr(result, "response_metadata", None) or {}
            usage = getattr(result, "usage_metadata", None) or {}
            generation.update(
                output=getattr(result, "content", result),
                model=meta.get("model_name") or meta.get("model"),
                usage_details={k: v for k, v in usage.items() if isinstance(v, int)},
                metadata={"finish_reason": meta.get("finish_reason")},
            )
            return result

    return RunnableLambda(call)
