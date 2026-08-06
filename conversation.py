import json
from llm import create_llm, generate_answer
from vector_store import retrieve_relevant_chunks
from reranker import create_reranker, score_documents, rank_scored_documents
from langchain_core.prompts import ChatPromptTemplate


def is_general_conversation(message):
    msg = message.lower().strip()

    exact_messages = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "salut", "bonjour", "bonsoir", "coucou",
        "سلام", "مرحبا", "أهلا", "اهلا", "صباح الخير", "مساء الخير", "عسلامة", "عسلامه",
    }

    if msg in exact_messages:
        return True

    identity_questions = [
        "who are you", "what are you", "what can you do", "introduce yourself",
        "qui es-tu", "qui êtes-vous", "que peux-tu faire", "que pouvez-vous faire", "présente-toi",
        "من أنت", "من انت", "ما اسمك", "ماذا تستطيع أن تفعل", "شنو تنجم تعمل", "شنو تعمل", "اش تعمل",
    ]

    for phrase in identity_questions:
        if phrase in msg:
            return True

    return False


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


def chat(message, history, memory_state, vector_store):
    llm = create_llm()

    if is_general_conversation(message):
        return "Hi! I’m here to help with Banque Centrale documents.", memory_state

    route = route_message(llm, message, memory_state)

    if route["intent"] == "GENERAL_CHAT":
        return "Hi! I’m here to help with Banque Centrale documents.", memory_state

    query_for_retrieval = route["rewrite_query"] or message

    retrieved_docs = retrieve_relevant_chunks(query_for_retrieval, vector_store)

    reranker = create_reranker()
    scored_docs = score_documents(reranker, query_for_retrieval, retrieved_docs)
    reranked_results = rank_scored_documents(scored_docs)
    documents_for_llm = [document for document, _ in reranked_results]

    memory_text = render_memory_state(memory_state)
    response = generate_answer(llm, message, documents_for_llm, memory_text)

    memory_state = update_memory_state(memory_state, route)
    return response, memory_state