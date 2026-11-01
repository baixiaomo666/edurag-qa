from langchain.chat_models import init_chat_model
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from base.config import config
from tools_lib import query_score_line

llm = init_chat_model(
            model=config.LLM_MODEL,
            model_provider="openai",
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            temperature=0.1  # 低温度，保证答案专业、稳定
        )


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

@tool
def add(a: int, b: int) -> str:
    """计算两个整数的和。当用户要求做加法时使用。"""
    print(f"    [工具被调用] add({a}, {b})")
    return str(a + b)

def agent_llm(state: State):
    llm_with_tools =llm.bind_tools([query_score_line])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def route(state: State):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END

builder = StateGraph(State)
builder.add_node("agent_llm", agent_llm)



def tools_node(state: State):
    last = state["messages"][-1]
    tool_map = {"query_score_line": query_score_line}
    results = []
    for tc in last.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

builder.add_node("tools", tools_node)
builder.add_edge(START, "agent_llm")
builder.add_conditional_edges(
    "agent_llm",
            route,
    {"tools": "tools",
                END: END})
builder.add_edge("tools", "agent_llm")          # ← 回边，没有它只调一次
graph = builder.compile()

if __name__ == "__main__":
    for q in ["清华大学  085400 2025年复试线多少？", "你好，介绍一下你自己"]:
        print(f"\n=== {q} ===")
        res = graph.invoke({"messages": [("user", q)]})
        for m in res["messages"]:
            print(f"  {type(m).__name__:12} tool_calls={getattr(m,'tool_calls',None)}  {str(m.content)[:50]}")