from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

class HallucinationSchema(BaseModel):
    """Binary score to check if the LLM generated response hallucinated"""

    binaryScore:str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

llm = ChatOpenAI(temperature=0)
structuredLLM = llm.with_structured_output(HallucinationSchema)

initialSystemMsg = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

chatPrompt = ChatPromptTemplate.from_messages([
    ("system", initialSystemMsg),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
])

hallucinationChain = chatPrompt | structuredLLM