from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from chains import generationChain, reflectionChain
from agentTrackerUtil import updateTracker, agentTracker

load_dotenv()

START = "start"
REFLECT = "reflect"
GENERATE = "generate"
CONDITIONAL_EDGE = "conditional_edge"
LAST = -1

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def shouldReflect(state: MessageGraph):
    conditionalEdgeResponse = REFLECT
    if len(state["messages"]) > 6:
        conditionalEdgeResponse = END
    updateTracker(f"{CONDITIONAL_EDGE}-{conditionalEdgeResponse}", state["messages"])
    return conditionalEdgeResponse


def generationNode(state: MessageGraph):
    latestResponse = {"messages": [generationChain.invoke({"messages": state["messages"]})]}
    updateTracker(GENERATE, state["messages"]+[latestResponse["messages"][0]])
    return latestResponse


def reflectionNode(state: MessageGraph):
    reflectionMsg = reflectionChain.invoke({"messages": state["messages"]})
    # Converting AI Message into Human Message to replicate human critique involvement
    latestResponse = {"messages": [HumanMessage(content=reflectionMsg.content)]}
    updateTracker(REFLECT, state["messages"]+[latestResponse["messages"][0]])
    return latestResponse


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generationNode)
builder.set_entry_point(GENERATE)
builder.add_node(REFLECT, reflectionNode)
builder.add_conditional_edges(GENERATE, shouldReflect, {
    END: END,
    REFLECT: REFLECT
})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

if __name__ == "__main__":
    print("Welcome to Reflection Agent!")
    inputs = HumanMessage(content="""Make this tweet better:"
                                @LangChainAI
        — newly Tool Calling feature is seriously underrated.

        After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

        Made a video covering their newest blog post

        """
    )
    updateTracker(START, [inputs])
    graphRes = graph.invoke({"messages": [inputs]})
    for trace in agentTracker:
        print("----------")
        print("Node Name:", trace.get("node"))
        print("Decision Node:", trace.get("decisionNode"))
        print("Last Message Type:", trace.get("lastMessageType"))
    print("----------")
    print("Content:", graphRes["messages"][-1].content, graphRes["messages"])
