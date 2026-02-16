from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from chains import generationChain, reflectionChain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"
LAST = -1

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def shouldReflect(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


def generationNode(state: MessageGraph):
    return {"messages": [generationChain.invoke({"messages": state["messages"]})]}


def reflectionNode(state: MessageGraph):
    reflectionMsg = reflectionChain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=reflectionMsg.content)]}


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
    graphRes = graph.invoke({"messages": [inputs]})
    print(graphRes)
