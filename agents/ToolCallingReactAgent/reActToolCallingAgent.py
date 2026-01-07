from dotenv import load_dotenv
from typing import List
from langchain.tools import tool
from langchain_classic.tools import BaseTool
from langchain.messages import HumanMessage, ToolMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from utils.callbacks import AgentCallbackHandler

load_dotenv()

@tool
def getTextLength(text:str) -> int:
    """Returns the length of the string (in characters)"""
    formattedText = text.strip("'\n").strip('"')
    return len(formattedText)


def findToolFromList(tools:List[BaseTool], toolName:str) -> BaseTool:
    for tool in tools:
        if tool.name == toolName:
            return tool
    raise ValueError(f"Tool {toolName} not found!")


if __name__ == "__main__":
    print("Welcome to ReAct Tool Calling Agent!")
    tools = [TavilySearch(), getTextLength]

    llm = ChatOpenAI(
        temperature=0,
        callbacks=[AgentCallbackHandler()]
    )

    llmWithTools = llm.bind_tools(tools)

    messages = [HumanMessage(content="Compare the temperature in Toronto (Canada) and Santorini (Greece)? Decribe in short answer!")]

    while True:
        aiMessage = llmWithTools.invoke(messages)
        toolCalls = getattr(aiMessage, "tool_calls", None) or []
        if len(toolCalls) > 0:
            messages.append(aiMessage)
            for toolCall in toolCalls:
                toolCallName = toolCall.get("name")
                toolCallArgs = toolCall.get("args", {})
                toolCallId = toolCall.get("id")

                toolToUse = findToolFromList(tools, toolCallName)
                observation = toolToUse.invoke(toolCallArgs)

                print(f"----- Observation -----\n", observation)

                toolMessage = ToolMessage(
                    content=str(observation),
                    tool_call_id=toolCallId
                )
                messages.append(toolMessage)
            
            continue
            
        print(f"----- Final Output -----\n", aiMessage.content)
        break