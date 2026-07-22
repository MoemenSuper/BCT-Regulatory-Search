from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
def create_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return llm

def generate_answer(llm,user_query,relevant_docs):

    context = [document.page_content for document in relevant_docs]
    context_string = "\n\n".join(context)

    prompt = ChatPromptTemplate.from_messages([
        ("system",

        "You answer questions about Tunisian regulatory documents. "
        "Use only the provided context. "
        "If the context does not contain the answer, say that the "
        "information was not found. Do not invent legal or regulatory facts."
        ),

        ("human",

        "Question: {user_query} \n\n\nContext: {context_string}"
        ),

        
    ])

    chain = prompt | llm
    return chain.invoke(
        {
        "user_query": user_query,
        "context_string": context_string
    }
    )
