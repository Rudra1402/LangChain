from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    Represents the state of the graph

    Attributes:
        question: question (Could be a web search query or if retrieved documents are relevant to the user query)
        generation: LLM generated response
        isWebSearchNeeded: Boolean
        documents: List of documents
    """

    question: str
    generation: str
    isWebSearchNeeded: bool
    documents: List[str]