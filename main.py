import os
from dotenv import load_dotenv

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain_core.messages import HumanMessage

load_dotenv()

tavilySearch = TavilySearch()

llm = ChatOpenAI(model="gpt-4")
tools = [tavilySearch]
reactPrompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=reactPrompt
)

agentExecutor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agentExecutor

def main():
    print("Hello from langchain-agents!")
    query = "Search 2 best and safest places near Canada to solo travel in 2026 with medium budget in CAD!"
    res = chain.invoke(
        input={
            "input": query
        }
    )
    print(res)


if __name__ == "__main__":
    main()
