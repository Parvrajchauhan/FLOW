from langchain_core.prompts import ChatPromptTemplate
from src.Backend.LLMs.geminiLLM import get_llm
from src.Backend.state.State import State
from typing import TypedDict, List, Optional
import json

llm = get_llm()

SYSTEM = """You are a task planner for a multi-modal AI agent. You will receive the user's query and a dictionary of uploaded files containg base64-encoded data and type.

Available tools that excutor node can call:
(extract_tool- extracts text from image or PDF files)
(audio_tool- transcribes audio files to text)
(youtube_tool-fetches transcript from a YouTube URL)

Available specialist nodes that excutor node can route to after tool execution:
(summarize_node-produces 1-line summary + 3 bullets + 5-sentence summary)
(code_node-Explain what code does, detect bugs, and mention time complexity.)
(sentiment_node-performs sentiment analysis and give Label + confidence + one-line justification.)
(cross_input_node-reasons across multiple sources that user upload and stored in the file registry)
(formatter_node-if no specialist node is needed, route here to format the final answer for the user)
(qa_node-to answer general query of user)

Your job is to create a plan for how to execute the user's request.

Rules:
- tool_sequence can be empty [] if no files uploaded and no URL detected
- specialist must be exactly one of the node names above
- if needs_clarification is true:
    - set follow_up_question
    - set tool_sequence to []
    - set specialist to null
- detect YouTube URLs in raw_text and return them in url
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "User query: {raw_text}\n\nFile registry: {file_registry}")
    ]
)


class PlannerOutput(TypedDict):
    tool_sequence: List[str]
    specialist: Optional[str]
    needs_clarification: bool
    follow_up_question: Optional[str]
    reasoning: str
    url: Optional[str]


planner_llm = llm.with_structured_output(
    PlannerOutput,
    include_raw=False
)


def planner_node(state: State) -> State:

    chain = PROMPT | planner_llm

    try:
        plan = chain.invoke(
            {
                "raw_text": state["raw_text"],
                "file_registry": json.dumps(
                    {
                        k: v["type"]
                        for k, v in state.get("file_registry", {}).items()
                    }
                ),
            }
        )

    except Exception as e:
        plan = {
            "tool_sequence": [],
            "specialist": "qa_node",
            "needs_clarification": False,
            "follow_up_question": None,
            "reasoning": f"Planner failed: {str(e)}",
            "url": None,
        }

    print("Planner Tool Sequence:", plan["tool_sequence"])
    print("Planner next task node:", plan["specialist"])
    print("Planner reasoning:", plan["reasoning"])

    return {
        "tool_sequence": plan.get("tool_sequence", []),
        "specialist": plan.get("specialist", "qa_node"),
        "needs_clarification": plan.get("needs_clarification", False),
        "follow_up_question": plan.get("follow_up_question"),
        "planner_reasoning": plan.get("reasoning", ""),
        "url": plan.get("url"),
        "plan_trace": [
            f"Planner: {plan.get('reasoning', '')}"
        ],
        "errors": [],
    }