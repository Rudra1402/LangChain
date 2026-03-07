from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

ragPrompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(temperature=0)

resGeneratorChain = ragPrompt | llm | StrOutputParser()