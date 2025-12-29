import os
from dotenv import load_dotenv

from typing import List
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

tavily = TavilyClient()
tavilySearch = TavilySearch()

class Source(BaseModel):
    """URL source that was searched on the web"""

    url:str = Field(description="URL source that was searched on the web")

class AgentResponse(BaseModel):
    """Agents response to the query"""

    answer:str = Field(description="Agents response to the query")
    sources:List[Source] = Field(default_factory=List, description="List of URL sources")

# Replacing this tool with TavilySearch from langchain_tavily when creating an agent
@tool
def webSearch(query: str):
    """Search the web with the given query and return the raw result string."""
    userInput = f"Searching web for <{query}>"
    print(userInput)
    return tavily.search(query=query)


llmModel = ChatOpenAI(model="gpt-5-nano", temperature=0)
tools = [tavilySearch]
# NOTE: create_agent() returns a runnable sequence like LCEL (LangChain Expression Language) using "|" pipe operator
agent = create_agent(model=llmModel, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-agents!")
    query = "Search 3 best and safest places near Canada to solo travel in 2026 with medium budget in CAD!"
    res = agent.invoke({
            "messages": HumanMessage(content=query)
    })
    print(res["messages"][-1].content)


if __name__ == "__main__":
    main()
