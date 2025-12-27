import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()


def webSearch(query: str):
    """Search the web with the given query and return the raw result string."""
    userInput = f"User Input - {query}"
    print(userInput)
    return userInput


llmModel = ChatOpenAI()
tools = [webSearch]
# NOTE: create_agent() returns a runnable sequence like LCEL (LangChain Expression Language) using "|" pipe operator
agent = create_agent(model=llmModel, tools=tools)


def main():
    print("Hello from langchain-agents!")
    res = agent.invoke({
            "messages": HumanMessage(content="What is the capital of France?")
    })
    print(res)


if __name__ == "__main__":
    main()
