from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm, tools

load_dotenv()

SYSTEM_PROMPT="""
You are a helpful assistant that can use tools to answer questions.
"""

def runReasoningEngine(state:MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    systemMsg = SystemMessage(content=SYSTEM_PROMPT)
    response = llm.invoke([systemMsg, *state["messages"]])
    return {"messages": [response]}

toolNodes = ToolNode(tools=tools)
