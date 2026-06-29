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
from headroom.integrations import HeadroomChatModel

MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5.4-mini")

class fib_state(TypedDict):
    n: int
    sum: int

def main():
    base_model = init_chat_model(
        model=MODEL,
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


if __name__ == "__main__":
    main()
