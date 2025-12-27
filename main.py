import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def webSearch(query: str):
    """Search the web with the given query and return the raw result string."""
    userInput = f"Searching web for <{query}>"
    print(userInput)
    return tavily.search(query=query)


llmModel = ChatOpenAI(model="gpt-5-nano")
tools = [webSearch]
# NOTE: create_agent() returns a runnable sequence like LCEL (LangChain Expression Language) using "|" pipe operator
agent = create_agent(model=llmModel, tools=tools)


def main():
    print("Hello from langchain-agents!")
    query = "Capital of Spain?"
    res = agent.invoke({
            "messages": HumanMessage(content=query)
    })
    print(res["messages"][-1].content)


if __name__ == "__main__":
    main()
