from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from nodes import runReasoningEngine, toolNodes

load_dotenv()

AGENT_REASON="agent_reason"
ACT="act"
LAST=-1

def shouldContinue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, runReasoningEngine)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, toolNodes)
flow.add_conditional_edges(AGENT_REASON, shouldContinue, {
    END: END,
    ACT: ACT
})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Welcome to ReAct Langgraph!")
    humanMessage = HumanMessage(content="Convert 50 USD to CAD and add 10 percent service tax to the converted amount!")
    res = app.invoke({"messages": [humanMessage]})
    print(res["messages"][LAST].content)
