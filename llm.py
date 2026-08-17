from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()
@lru_cache(maxsize=1)
def create_llm():
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        reasoning_effort="medium",
        reasoning_format="hidden",
        max_tokens=2048,
    )

    return llm

def generate_answer(llm,user_query,relevant_docs,memory_summary=""):

    context = [document.page_content for document in relevant_docs]
    context_metadata = [str(document.metadata) for document in relevant_docs]
    #Convert the context and its metadata into 1 string respectively, so the LLM can use it 
    context_string = "\n\n".join(context)
    context_metadata_string = "\n\n".join(context_metadata)

    prompt = ChatPromptTemplate.from_messages([
        ("system",

        """You answer questions about Tunisian regulatory documents. 
        Use only the provided context. 
        The Conversation Summary is only to understand follow-up questions and references such as:
        - 'it'
        - 'that circular'
        - 'its publication date'
        - 'the previous one'

        Do not use the Conversation Summary as factual evidence.
        Never answer using the summary alone.

        If the context does not contain the answer, say that the 
        information was not found. Do not invent legal or regulatory facts.
        At the end, cite the exact source filename and page number from the provided Metadata. And if no information is found. do not cite any filename or pagenumber.
        Never write 'document.pdf' or placeholder text."""
        ),

        ("human",

        f"Conversation memory:\n{memory_summary}\n\n"
        "Question: {user_query} \n\n\nContext: {context_string} \t\t Metadata: {context_metadata_string}"
        ),

        
    ])

    chain = prompt | llm
    return chain.invoke(
        {
        "memory_summary": memory_summary,
        "user_query": user_query,
        "context_string": context_string,
        "context_metadata_string": context_metadata_string
    }
    ).content
