from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.Backend.tools.extract_tool import extract_tool
from src.Backend.tools.audio_tool import audio_tool
from src.Backend.tools.YouTube_tool import youtube_tool

from src.Backend.Nodes.ingest_node import ingest_node
from src.Backend.Nodes.planner_node import planner_node
from src.Backend.Nodes.excutor_node import executor_node
from src.Backend.Nodes.clarity_node import clarify_node
from src.Backend.Nodes.format_node import formatter_node
from src.Backend.Nodes.summarize_node import summarize_node
from src.Backend.Nodes.code_node import code_node
from src.Backend.Nodes.sentiment_node import sentiment_node
from src.Backend.Nodes.cross_input_node import cross_input_node
from src.Backend.state.State import State

from .conditional_edges import route_after_planner, route_after_executor

tools = [extract_tool, audio_tool, youtube_tool]

builder = StateGraph(State)

# nodes
builder.add_node("ingest",      ingest_node)
builder.add_node("planner",     planner_node)
builder.add_node("clarify",     clarify_node)
builder.add_node("executor",    executor_node)
builder.add_node("tools",       ToolNode(tools))
builder.add_node("summarize",   summarize_node)
builder.add_node("sentiment",   sentiment_node)
builder.add_node("code",        code_node)
builder.add_node("cross_input", cross_input_node)
builder.add_node("formatter",   formatter_node)

# edges
builder.add_edge(START,      "ingest")
builder.add_edge("ingest",   "planner")

builder.add_conditional_edges("planner", route_after_planner, {
    "clarify":           "clarify",
    "executor":          "executor",
})

builder.add_conditional_edges("executor", route_after_executor, {
    "tools":             "tools",
    "summarize_node":    "summarize",
    "executor":          "executor",
    "cross_input_node":  "cross_input",
    "code_node":        "code",
    "sentiment_node":   "sentiment",
    "formatter_node":    "formatter",
})

builder.add_edge("tools",    "executor") 
builder.add_edge("clarify",     "formatter")
builder.add_edge("summarize",   "formatter")
builder.add_edge("sentiment",   "formatter")
builder.add_edge("code",        "formatter")
builder.add_edge("cross_input", "formatter")
builder.add_edge("formatter",   END)

graph = builder.compile()

def get_graph():
    return graph