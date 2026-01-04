from dotenv import load_dotenv
from typing import List, Union
from langchain.tools import tool
from langchain_classic.tools import Tool
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

@tool
def getTextLength(text:str) -> int:
    """Returns the length of the string (in characters)"""
    formattedText = text.strip("'\n").strip('"')
    return len(formattedText)


def findToolFromList(tools:List[Tool], toolName:str) -> Tool:
    for tool in tools:
        if tool.name == toolName:
            return tool
    raise ValueError(f"Tool {toolName} not found!")


if __name__ == "__main__":
    print("Welcome to Custom LangChain Agent!")
    tools = [getTextLength]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools), tool_names=",".join([t.name for t in tools]))

    llm = ChatOpenAI(temperature=0, stop=["/Observation", "Observation", "Observation:"])

    llmContext = []

    agent = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
    } | prompt | llm | ReActSingleInputOutputParser()

    agentStep: Union[AgentAction, AgentFinish] = agent.invoke({
        "input": "What is the text length of Oesophagous in characters?",
        "agent_scratchpad": llmContext
    })

    if isinstance(agentStep, AgentAction):
        toolName = agentStep.tool
        toolToUse = findToolFromList(tools, toolName)
        toolInput = agentStep.tool_input
        observation = toolToUse.func(toolInput)
        llmContext.append((agentStep, str(observation)))

    agentStep: Union[AgentAction, AgentFinish] = agent.invoke({
        "input": "What is the text length of Oesophagous in characters?",
        "agent_scratchpad": llmContext
    })

    if isinstance(agentStep, AgentFinish):
        print(agentStep.return_values)