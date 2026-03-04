from typing import Any, Dict
from graph.state import GraphState
from ingestion import retrieverStore

def retrieverNode(state: GraphState) -> Dict[str, Any]:
    question = state['question']
    retrieverRes = retrieverStore.invoke(question)
    return {
        "documents": retrieverRes,
        "question": question
    }
