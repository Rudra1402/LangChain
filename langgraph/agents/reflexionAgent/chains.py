import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from schemas import ResponseLLM

llm = ChatOpenAI(model="gpt-4-turbo-preview")
parserJson = JsonOutputToolsParser(return_id=True)
parserPydantic = PydanticToolsParser(tools=[ResponseLLM])

actorPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are expert researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format.")
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

firstResponderPromptTemplate = actorPromptTemplate.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

firstResponderChain = firstResponderPromptTemplate | llm.bind_tools(
    tools=[ResponseLLM], tool_choice="ResponseLLM"
)

if __name__ == "__main__":
    humanMsg = HumanMessage(content="Write about an AI powered SOC / autonomous SOC problem domain list startups that do that and their raised capital")
    chain = (
        firstResponderPromptTemplate |
        llm.bind_tools(
            tools=[ResponseLLM], tool_choice="ResponseLLM"
        ) |
        parserPydantic
    )
    res = chain.invoke({"messages": [humanMsg]})
    print(res)