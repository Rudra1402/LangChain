import os
from dotenv import load_dotenv

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tavilySearch = TavilySearch()

llm = ChatOpenAI(model="gpt-4")
tools = [tavilySearch]
reactPrompt = hub.pull("hwchase17/react")

outputParser = PydanticOutputParser(pydantic_object=AgentResponse)
reactPromptWithFormatInstructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"]
).partial(format_instructions=outputParser.get_format_instructions())

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=reactPromptWithFormatInstructions
)

agentExecutor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extractOutput = RunnableLambda(lambda x: x["output"]) # Only extract the "output" proprty from AgentExecutor response
parseOutput = RunnableLambda(lambda x: outputParser.parse(x)) # Parse JSON string for "output" property into AgentResponse format
chain = agentExecutor | extractOutput | parseOutput

def main():
    print("Hello from langchain-agents!")
    query = "What are the best veg food options in Barcelona?"
    res = chain.invoke(
        input={
            "input": query
        }
    )
    print(res)


if __name__ == "__main__":
    main()
