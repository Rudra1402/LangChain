# type: ignore

from dotenv import load_dotenv
from .constants import RETRIEVE, GENERATE, GRADEDOCS, WEBSEARCH
from .nodes import retrieverNode, gradeDocuments, webSearchNode, generatorNode
from .state import GraphState
from langgraph.graph import StateGraph, END, START

load_dotenv()

def conditionalEdgeCheck(state: GraphState):
    if state['isWebSearchNeeded']:
        return WEBSEARCH
    return GENERATE

graphInit = StateGraph(GraphState)
graphInit.add_node(RETRIEVE, retrieverNode)
graphInit.add_node(GRADEDOCS, gradeDocuments)
graphInit.add_node(WEBSEARCH, webSearchNode)
graphInit.add_node(GENERATE, generatorNode)

graphInit.add_edge(START, RETRIEVE)
graphInit.add_edge(RETRIEVE, GRADEDOCS)
graphInit.add_conditional_edges(GRADEDOCS, conditionalEdgeCheck, {
    WEBSEARCH: WEBSEARCH,
    GENERATE: GENERATE
})
graphInit.add_edge(WEBSEARCH, GENERATE)
graphInit.add_edge(GENERATE, END)

app = graphInit.compile()
print(app.get_graph().draw_mermaid())