from typing import Dict, Any
from graph.chains.finalResponseGenerator import resGeneratorChain
from graph.state import GraphState

def generatorNode(state: GraphState) -> Dict[str, Any]:
    question = state['question']
    documents = state['documents']

    res = resGeneratorChain.invoke({"question": question, "context": documents})
    return {
        "question": question,
        "documents": documents,
        "generation": res
    }