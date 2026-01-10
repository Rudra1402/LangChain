import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorStore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings
)
retriever = vectorStore.as_retriever(search_kwargs={"k":2})

promptTemplate = ChatPromptTemplate.from_template(
    """Answer the following question based on the given context only:
    {context}
    Question: {question}
    """
)

def formatDocs(docs):
    """Stringify the Document objects from Pinecone"""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieveWithoutLCEL(query:str):
    """
    Simple retrieval chain without using LCEL
    Manually retrieves, formats, generates response
    """
    retrievedDocs = retriever.invoke(query)
    docs = formatDocs(retrievedDocs)
    messages = promptTemplate.format_messages(context=docs, question=query)
    llmRes = llm.invoke(messages)
    return llmRes.content

def runnableChainLCEL():
    chain = (
        RunnablePassthrough.assign(
            context = (lambda x: x["question"]) |
                retriever |
                formatDocs
        ) |
        promptTemplate |
        llm |
        StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    print("RAG In Process...")
    query = "Are there any stocks or ETFs listed? If yes, list them!"

    # [ Without LCEL ]
    # answer = retrieveWithoutLCEL(query)

    # [ With LCEL ]
    chain = runnableChainLCEL()
    answer = chain.invoke({"question": query})
    
    print(answer)