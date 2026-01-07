## LangChain

<img width="350" height="200" alt="langchain" src="https://github.com/user-attachments/assets/94d807bc-2fc7-4269-b55e-25dfd5e4eb6b" />

---

Agent implementations using different approaches to learn about LangChain's evolution. Initially, started with LangChain v1 by using `create_agent()` which uses LangGraph under the hood, then moved a step back to the previous version which uses `create_react_agent()` & `AgentExecutor` that has ReAct agent implementation using the famous ReAct prompt template (Thought-Action-Observation). Again moved a step back and implemented `create_react_agent()` from scratch using LangChain Expression Language (LCEL) to understand the core working of `create_react_agent()`, again using the ReAct prompt template (Thought-Action-Observation). Then moved to LLM's Tool Calling technique which got rid of the ReAct prompt template and the LCEL, this involves binding tools to llm provider using `.bind_tools()`.

## 🤖 Breakdown
- `textSummarizer.py` - [ LCEL ] Simple QnA chatbot that answers user queries related to the data provided. (Note: This is not a RAG chatbot. I am not chunking or vectorizing data. This approach is not suitable for large data). This was implemented to learn basics of LangChain and LCEL approach.
  - Packages
    - langchain-core
    - langchain-openai

- `searchAgentUsingCreateAgent.py` - [ create_agent() ] Web search agent that answers user queries by making web searches and returns answers along with sources. This agent is suitable for questions that can be found on the internet. Removes the hassle of manually going through the internet to find some answers. This agent finds the answer and gives the sources too. The `create_agent()` method has LangGraph implementation under the hood.
  - Packages
    - langchain
    - langchain-core
    - langchain-openai
    - langchain-tavily (Tavily provides Web Search services using its API. This is the LangChain wrapper for Tavily.)
    - pydantic
