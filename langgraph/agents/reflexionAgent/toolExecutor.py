from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from schemas import ResponseLLM, ResponseRevisedLLM
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from typing import List

load_dotenv()

tavilySearchTool = TavilySearch(max_results=4)

def runQueries(searchQueries:List[str], **kwargs):
    """Run the generated queries"""
    return tavilySearchTool.batch([{"query": query} for query in searchQueries])

tools = ToolNode(
    [
        StructuredTool.from_function(runQueries, name=ResponseLLM.__name__),
        StructuredTool.from_function(runQueries, name=ResponseRevisedLLM.__name__)
    ]
)