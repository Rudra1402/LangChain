import asyncio
from dotenv import load_dotenv
from langchain_community.document_loaders import weather
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

load_dotenv()

llm = ChatOpenAI()

# This is for MCP Server with transport "stdio"
serverParams = StdioServerParameters(
    command="python",
    args=["/Users/rudrapatel/Desktop/Projects/langchain-agents/mcp/servers/mathServer.py"]
)

async def getMathToolsSession():
    async with stdio_client(serverParams) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            return session
    

async def getWeatherTools():
    async with sse_client("http://127.0.0.1:8000/sse") as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("Weather Session Initialized!")
            tools = await load_mcp_tools(session)
            return tools
    

async def main():
    async with stdio_client(serverParams) as (math_read, math_write):
        async with sse_client("http://127.0.0.1:8000/sse") as (weather_read, weather_write):
            async with ClientSession(read_stream=math_read, write_stream=math_write) as math_session:
                async with ClientSession(read_stream=weather_read, write_stream=weather_write) as weather_session:
                    
                    await math_session.initialize()
                    print("✅ Math Session Initialized!")
                    
                    await weather_session.initialize()
                    print("✅ Weather Session Initialized!")
                    
                    mathTools = await load_mcp_tools(math_session)
                    weatherTools = await load_mcp_tools(weather_session)
                    tools = mathTools + weatherTools
                    
                    print(f"📦 Available tools: {[t.name for t in tools]}")
                    
                    agent = create_agent(model=llm, tools=tools)
                    
                    res = await agent.ainvoke({
                        "messages": [HumanMessage(content="What is the weather in Toronto?")]
                    })

                    print(res["messages"][-1].content)

                    toolCallsMade = []
                    for toolMsg in res["messages"]:
                        if isinstance(toolMsg, ToolMessage):
                            toolCallsMade.append(
                                {
                                    "name": toolMsg.name,
                                    "callId": toolMsg.tool_call_id
                                }
                            )
                    print(toolCallsMade)

if __name__ == "__main__":
    asyncio.run(main())