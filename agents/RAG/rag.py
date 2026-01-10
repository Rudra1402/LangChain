import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

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

if __name__ == "__main__":
    print("RAG In Process...")
    query = "Are there any stocks or ETFs listed? If yes, list them!"
    answer = retrieveWithoutLCEL(query)
    print(answer)