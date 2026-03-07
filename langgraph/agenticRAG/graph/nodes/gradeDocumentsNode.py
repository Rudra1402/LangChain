from typing import Dict, Any
from graph.state import GraphState
from graph.chains.retrievalDocsGrader import gradeResultsChain

def gradeDocuments(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    question = state["question"]
    documents = state["documents"]

    filteredDocs = []
    isWebSearchNeeded = False

    for doc in documents:
        docRes = gradeResultsChain.invoke({
            "question": question,
            "document": doc.page_content # type: ignore
        })

        docGrade = docRes.binaryScore # type: ignore
        if docGrade.lower() == "yes":
            filteredDocs.append(doc)
        else:
            isWebSearchNeeded = True
            continue
    
    return {
        "documents": filteredDocs,
        "question": question,
        "isWebSearchNeeded": isWebSearchNeeded
    }
