from langchain_groq import ChatGroq


def create_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return llm
