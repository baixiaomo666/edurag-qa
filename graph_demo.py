from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph, add_messages

from base.config import config
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.retrieval.bm25_search import BM25Search
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.query_classifier import QueryClassifier
from rag_qa.core.strategy_selector import StrategySelector
from tools_lib import query_score_line

bm25 = BM25Search()
query_classifier=QueryClassifier()
strategy_selector=StrategySelector()
rag_prompts=RAGPrompts()
mysql_client = MySQLClient()
llm = init_chat_model(
            model=config.LLM_MODEL,
            model_provider="openai",
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            temperature=0.1  # 低温度，保证答案专业、稳定
        )

class State(TypedDict):
    query: str
    faq_hit: bool
    faq_answer: str
    answer: str
    context: str
    category: str
    history: str
    session_id: str
    messages: Annotated[list[AnyMessage], add_messages]



def faq_search(state: State):

    ans, need_rag = bm25.search(state["query"])  # 它的返回值本来就是两个
    return {"faq_hit": not need_rag, "faq_answer": ans or ""}  # 自己组 dict


def faq_answer(state: State):              # 命中：直出（落库交给 save 节点）
    print("→ 走 FAQ 直出")
    return {"answer": state["faq_answer"]}


def generate(state: State):

    prompt = RAGPrompts.rag_prompt().format(
        context=state["context"],
        question=state["query"],
        history=state["history"],
        phone=config.CUSTOMER_SERVICE_PHONE,
    )

    return {"answer": "".join(c.content for c in llm.stream(prompt) if c.content)}


def route_after_faq(state: State):         # ← 边：只读 State，返回节点名
    return "faq_answer" if state["faq_hit"] else "classify"


def route_after_classify(state: State):         # ← 边：只读 State，返回节点名
    return "generate" if state["category"] == "通用知识" else "retrieve"


def classify(state):
    category = query_classifier.predict_category(state["query"])
    if category == "通用知识":
        return {
            "category": category,
            "context": "（本次回答无需检索知识库，请直接依据通用常识作答，不要以“信息不足”为由拒答）",
        }
    return {"category": category}


def retrieve(state: State):
    documents = strategy_selector.strategy_select(state["query"])
    return {"context": "\n\n".join(d.page_content for d in documents)}

def load_history(state: State):
    if not state["session_id"]:
        return {"history": ""}
    rows = mysql_client.fetch_recent_history(state["session_id"])
    return {"history": "\n".join(f"Q: {r['question']}\nA: {r['answer']}" for r in rows)}


def save(state: State):
    if state.get("session_id") and state.get("answer"):
        mysql_client.update_session_history(state["session_id"], state["query"], state["answer"])
    return {}

def agent_llm(state: State):
    llm_with_tools = llm.bind_tools([query_score_line])

    if state.get("messages"):
        msgs = state["messages"]        # 后续轮次：完整历史
        to_write = []
    else:
        msgs = [HumanMessage(content=RAGPrompts.rag_prompt().format(
            context=state["context"], question=state["query"],
            history=state["history"], phone=config.CUSTOMER_SERVICE_PHONE))]
        to_write = msgs                 # ★ 第一次：把 Human 也一起返回

    resp = llm_with_tools.invoke(msgs)

    out = {"messages": to_write + [resp]}
    if not resp.tool_calls:
        out["answer"] = resp.content
    return out

def search_tool(state: State):
    last = state["messages"][-1]
    tool_map = {"query_score_line": query_score_line}
    results = []
    for tc in last.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

def route_after_retrieve(state: State):
    print(">>> route 进来了, messages =", state.get("messages"))
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "search_tool"
    return "save"

builder = StateGraph(State)
builder.add_node("load_history", load_history)
builder.add_node("faq_search", faq_search)
builder.add_node("faq_answer", faq_answer)
builder.add_node("retrieve", retrieve)
builder.add_node("classify", classify)
builder.add_node("generate", generate)
builder.add_node("agent_llm", agent_llm)
builder.add_node("search_tool", search_tool)
builder.add_node("save", save)


builder.add_edge(START, "load_history")
builder.add_edge("load_history", "faq_search")
builder.add_conditional_edges("faq_search", route_after_faq)
builder.add_conditional_edges("classify", route_after_classify)
builder.add_edge("retrieve", "agent_llm")
builder.add_conditional_edges("agent_llm",
                              route_after_retrieve,
                        {"search_tool": "search_tool",
                                   "save": "save"})
builder.add_edge("search_tool", "agent_llm")
builder.add_edge("faq_answer", "save")
builder.add_edge("generate", "save")
builder.add_edge("save", END)

graph = builder.compile()

SID = "test-002"

r = graph.invoke({"query": "北京大学 085400 2025 复试线多少？", "faq_hit": False,
                   "faq_answer": "", "answer": "", "context": "", "category": "",
                   "history": "", "session_id": SID})
print("history:", repr(r["history"]))
print("answer:", r["answer"][:60])