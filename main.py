import os
from dotenv import load_dotenv
load_dotenv()
from typing import Annotated, TypedDict

# langchain imports
from langchain.tools import tool
from langchain.messages import AIMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

# langgraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# headroom-ai imports
from headroom.integrations.langchain import HeadroomChatModel

MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5.4-mini")

class fibsum_state(TypedDict):
    n: int
    fact: list[int]
    sum: int

def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

# 3 -> fib(2)+fib(1) -> 1+1 = 2

workflow = StateGraph(fibsum_state)


def main():
    base_model = init_chat_model(
        model=MODEL,
        temperature=0.6,
    )
    model = HeadroomChatModel(
        wrapped_model=base_model,
    )
    agent = create_agent(
        model=model,
        tools=[],
    )

    response = agent.invoke({"messages": [HumanMessage(content="Hello, how are you?")]})
    print(response["messages"][-1].content)
    print(model.get_savings_summary())


if __name__ == "__main__":
    main()
