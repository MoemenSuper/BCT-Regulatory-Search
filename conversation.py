from llm import create_llm,generate_answer
from vector_store import retrieve_relevant_chunks
from reranker import create_reranker,score_documents,rank_scored_documents
from langchain_core.prompts import ChatPromptTemplate

def is_general_conversation(message):
    msg = message.lower().strip()

    exact_messages = {
        # English
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening",

        # French
        "salut", "bonjour", "bonsoir", "coucou",

        # Arabic
        "سلام",
        "مرحبا",
        "أهلا",
        "اهلا",
        "صباح الخير",
        "مساء الخير",
        "عسلامة",
        "عسلامه",
    }

    if msg in exact_messages:
        return True

    identity_questions = [
        # English
        "who are you",
        "what are you",
        "what can you do",
        "introduce yourself",

        # French
        "qui es-tu",
        "qui êtes-vous",
        "que peux-tu faire",
        "que pouvez-vous faire",
        "présente-toi",

        # Arabic
        "من أنت",
        "من انت",
        "ما اسمك",
        "ماذا تستطيع أن تفعل",
        "شنو تنجم تعمل",
        "شنو تعمل",
        "اش تعمل",
    ]

    for phrase in identity_questions:
        if phrase in msg:
            return True

    return False


def chat(message, history, memory_summary, vector_store):
    if is_general_conversation(message):
        return "Hi! I’m here to help with Banque Centrale documents.", memory_summary
    
    # 2) retrieve docs for the current message
    query_for_retrieval = message   # retrieval uses ONLY the current message
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
    print("\n===== MEMORY SUMMARY =====")
    print(memory_summary)
    print("==========================\n\n")
    response = generate_answer(llm, message, documents_for_llm, memory_summary)
    response_text = response.content if hasattr(response, "content") else str(response)

    # 5) update memory_summary
    memory_summary = update_memory_summary(llm, memory_summary, message, response_text)
    # 6) return answer + memory_summary
    return response, memory_summary



def update_memory_summary(llm, memory_summary, message, response_text):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You update a short memory summary for a Banque Centrale regulatory search assistant. "
         "Keep it under 80 words. Store only: the user's goal, the current topic, "
         "important circular numbers, entities, and open follow-up questions. "
         "Do not copy the full dialogue."),
        ("human",
         "Existing summary:\n{memory_summary}\n\n"
         "New user message:\n{message}\n\n"
         "Assistant answer:\n{response_text}\n\n"
         "Return only the updated summary.")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "memory_summary": memory_summary,
        "message": message,
        "response_text": response_text,
    })
    return result.content.strip()