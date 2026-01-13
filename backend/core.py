import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage, HumanMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddingModel = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512
)

vectorStore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"],
    embedding=embeddingModel
)

model = init_chat_model("gpt-5", model_provider="openai")

@tool(response_format="content_and_artifact")
def retrieveContext(query:str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    
    fetchedDocs = vectorStore.as_retriever().invoke(query, k=2)
    formattedDocs = "\n\n".join(
        (f"Source: {doc.metadata.get("source", "Unknown")}\n\nContent: {doc.page_content}")
        for doc in fetchedDocs
    )

    return formattedDocs, fetchedDocs

def runLLM(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    systemPrompt = (
        "You are a helpful AI assistant that answers questions about Stripe API documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(
        model=model,
        system_prompt=systemPrompt,
        tools=[retrieveContext]
    )

    messages = [HumanMessage(content=query)]

    agentRes = agent.invoke({"messages": messages})

    answer = agentRes["messages"][-1].content

    sourceDocs = []
    for msg in agentRes["messages"]:
        if isinstance(msg, ToolMessage) and hasattr(msg, "artifact"):
            if isinstance(msg.artifact, list):
                sourceDocs.extend(msg.artifact)

    return {
        "answer": answer,
        "context": sourceDocs
    }

if __name__ == "__main__":
    print("Welcome to Documentation Assitant!")
    result = runLLM(query="What is Auto-Pagination?")
    print(result)