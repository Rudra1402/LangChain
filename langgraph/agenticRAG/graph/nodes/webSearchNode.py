from dotenv import load_dotenv
from typing import Dict, Any
from langchain_classic.schema import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

load_dotenv()

webSearchTool = TavilySearch(max_results=3)

def webSearchNode(state: GraphState)-> Dict[str, Any]:
    question = state["question"]
    documents = state["documents"]
    searchRes = webSearchTool.invoke({"query": question})
    combinedSearchContent = "\n".join(
        [searchItem["content"] for searchItem in searchRes["results"]]
    )
    searchDocument = Document(page_content=combinedSearchContent)
    if documents is not None:
        documents.append(searchDocument) # type: ignore
    else:
        documents = [searchDocument]
    
    return {
        "documents": documents,
        "question": question
    }