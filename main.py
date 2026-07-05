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
    target_n: int
    curr_n: int
    fact: dict[int, int]

def fib_node(state: fibsum_state) -> int:
    # put into variables for readability
    n = state["curr_n"]
    fact = state["fact"]

    if n <= 1:
        return fact[n]
    if fact[n - 1] and fact[n - 2]:
        return fact[n - 1] + fact[n - 2]
    else:
        return "Go down"

def router(state: fibsum_state) -> int:
    return

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
