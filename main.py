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

def fib(n: int, fact: dict[int, int]) -> int:
    if n <= 1:
        return fact[n]
    if fact[n - 1] != -1 and fact[n - 2] != -1:
        fact[n] = fact[n - 1] + fact[n - 2]
    else:
        fact[n] = -1
    return fact[n]

def calc_node(state: fibsum_state):
    curr = state["curr_n"]
    fact = state["fact"]
    fib(curr, fact)
    return {"fact": fact}

def up_node(state: fibsum_state):
    return {"curr_n": state["curr_n"] + 1}

def down_node(state: fibsum_state):
    return {"curr_n": state["curr_n"] - 1}

def router(state: fibsum_state) -> str:
    n = state["target_n"]
    curr = state["curr_n"]
    result = state["fact"][curr]

    if result == -1:
        return "Down"
    if curr < n:
        return "Up"
    return "Done"

workflow = StateGraph(fibsum_state)

workflow.add_node("calc", calc_node)
workflow.add_node("up", up_node)
workflow.add_node("down", down_node)

workflow.add_edge(START, "calc")
workflow.add_conditional_edges("calc", router, {"Up": "up", "Down": "down", "Done": END})
workflow.add_edge("up", "calc")
workflow.add_edge("down", "calc")

fib_app = workflow.compile()


def build_fact_dict(n: int) -> dict[int, int]:
    fact = {i: -1 for i in range(n + 1)}
    fact[0] = 0
    if n >= 1:
        fact[1] = 1
    return fact


@tool
def compute_fibonacci(target_n: int) -> str:
    """Compute the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1)."""
    if target_n < 0:
        return "Error: target_n must be non-negative"

    final_state = fib_app.invoke({
        "target_n": target_n,
        "curr_n": target_n,
        "fact": build_fact_dict(target_n),
    })
    result = final_state["fact"][target_n]
    return f"fib({target_n}) = {result}"


def main():
    base_model = init_chat_model(
        model=MODEL,
        temperature=0.2,
    )
    model = HeadroomChatModel(
        wrapped_model=base_model,
    )
    agent = create_agent(
        model=model,
        tools=[compute_fibonacci],
    )

    response = agent.invoke({
        "messages": [HumanMessage(content="What is the 10th Fibonacci number?")],
    })
    print(response["messages"][-1].content)
    print(model.get_savings_summary())


if __name__ == "__main__":
    main()
