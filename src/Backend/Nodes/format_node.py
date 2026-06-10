from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from src.Backend.state.State import State


def _extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(block.get("text", "") for block in content
                                              if isinstance(block, dict) and block.get("type") == "text").strip()
    return ""


def _get_final_answer(messages: list) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage): continue
        
        text = _extract_text_from_content(msg.content)
        if not text: continue
        
        if text.startswith("[PLANNER]"): continue
        
        if msg.tool_calls and not text: continue
        
        return text
    return "No response generated."


def _truncate(text: str, limit: int) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def _summarise_invoke(messages: list) -> str:
    sections = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = _extract_text_from_content(msg.content)
            if text:
                sections.append(f"USER: {_truncate(text, 120)}")
            break

    planner_steps = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            text = _extract_text_from_content(msg.content)
            if text.startswith("[PLANNER]"):
                planner_steps.append(_truncate(text.removeprefix("[PLANNER]").strip(), 200))

    if planner_steps:
        sections.append("PLAN:\n" + "\n".join(f"  • {s}" for s in planner_steps))

    tool_result_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            result_text = _extract_text_from_content(msg.content)
            tool_result_map[msg.tool_call_id] = _truncate(result_text, 80)

    tool_log = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                result_snippet = tool_result_map.get(tc["id"], "pending")
                args_repr = ", ".join(f"{k}={_truncate(str(v), 40)!r}" for k, v in tc["args"].items())
                tool_log.append(f"  • {tc['name']}({args_repr}) → {result_snippet}")

    if tool_log:
        sections.append("TOOLS:\n" + "\n".join(tool_log))

    final = _get_final_answer(messages)
    if final and final != "No response generated.":
        sections.append(f"OUTCOME: {_truncate(final, 150)}")

    return "\n\n".join(sections) if sections else "Empty invoke."

def formatter_node(state: State) -> dict:
    messages = state.get("messages", [])
    plan_trace = state.get("plan_trace", [])

    extracted_texts: dict = dict(state.get("extracted_texts", {}))

    audio_transcript = state.get("audio_transcript", "")
    if audio_transcript:
        extracted_texts["audio_transcript"] = _truncate(audio_transcript, 300)

    yt_transcript = state.get("yt_transcript", "")
    if yt_transcript:
        extracted_texts["yt_transcript"] = _truncate(yt_transcript, 300)

    ocr_confidences = state.get("ocr_confidences", {})
    if ocr_confidences:
        extracted_texts["ocr_confidences"] = ocr_confidences

    final_response = _get_final_answer(messages)

    if final_response == "No response generated." and extracted_texts:
        file_texts = []

        for key, value in extracted_texts.items():
            if key == "ocr_confidences":
                continue

            if isinstance(value, str) and value.strip():
                file_texts.append(f" {key} \n{_truncate(value, 3000)}")

        final_response = "Content extracted successfully. See extracted content panel."
    

    tools_used = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append({
                    "tool": tc["name"],
                    "args": list(tc["args"].keys()),
                })

        if isinstance(msg, ToolMessage):
            if tools_used:
                tools_used[-1]["status"] = "success"

    invoke_summary = _summarise_invoke(messages)

    formatted_output = {
        "final_response": final_response,
        "plan_trace": plan_trace,
        "tools_used": tools_used,
        "extracted_texts": extracted_texts,
        "errors": state.get("errors", []),
    }

    for key, value in formatted_output.items():
        print(f"key: {key}\nvalue: {value}")

    return {
        "final_response": formatted_output,
        "invoke_summary": invoke_summary,
        "plan_trace": ["formatted_output"],
    }