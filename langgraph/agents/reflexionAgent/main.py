from dotenv import load_dotenv
from typing import Literal
from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from chains import firstResponderChain, revisedResponseChain
from toolExecutor import tools

load_dotenv()

FIRST_RESPONDER = "first_responder"
SECOND_RESPONDER = "second_responder"
TOOLS = "tools"
MAX_ITERS = 2

def firstResponderNode(state: MessagesState):
    """Draft the initial response"""
    result = firstResponderChain.invoke({"messages": state["messages"]})
    return {"messages": [result]}

def revisedResponderNode(state: MessagesState):
    """Draft the revised response"""
    result = revisedResponseChain.invoke({"messages": state["messages"]})
    return {"messages": [result]}

def revisedResponseEventLoop(state: MessagesState):
    """Determine whether to continue revising the response or end the loop"""
    toolExecCount = 0
    for message in state["messages"]:
        if isinstance(message, ToolMessage):
            toolExecCount += 1
    if toolExecCount > MAX_ITERS:
        return END
    return "tools"

graph = StateGraph(MessagesState)
graph.add_node(FIRST_RESPONDER, firstResponderNode)
graph.add_node(SECOND_RESPONDER, revisedResponderNode)
graph.add_node(TOOLS, tools)
graph.add_edge(START, FIRST_RESPONDER)
graph.add_edge(FIRST_RESPONDER, TOOLS)
graph.add_edge(TOOLS, SECOND_RESPONDER)
graph.add_conditional_edges(SECOND_RESPONDER, revisedResponseEventLoop, {
    END: END,
    TOOLS: TOOLS
})

compiledGraph = graph.compile()
print(compiledGraph.get_graph().draw_mermaid())


if __name__ == "__main__":
    print("Welcome to Reflexion Agent!")
    humanMessage = HumanMessage(content="Write about an AI powered SOC / autonomous SOC problem domain list startups that do that and their raised capital")
    res = compiledGraph.invoke({"messages": [humanMessage]})
    lastMsg = res["messages"][-1]
    if isinstance(lastMsg, AIMessage):
        print(lastMsg.tool_calls[0]["args"]["answer"])