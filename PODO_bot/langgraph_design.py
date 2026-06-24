import operator
import os
from typing import Annotated, TypedDict, Sequence, Any, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

# LLM 초기화 (Ollama 로컬 모델)
# num_predict=200을 설정하여 모델이 최대 200토큰까지만 생성하도록 강제 제한 (무한루프 방지)
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0, num_predict=200)

# 1. State (Memory)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task_info: str
    image_size: tuple
    plan: Optional[str]
    human_approved: Optional[bool]
    route_to: str  # 라우팅을 결정할 플래그 ("END", "planner" 등)
    image_paths: Annotated[list[str], operator.add]  # 누적할 이미지 경로 리스트
    final_image: Optional[str]  # summary_agent가 반환할 최종 이미지 경로
    error_info: Optional[str]
    failed_agent: Optional[str]

# 임시 툴(Tool) 정의
def resize_tool():
    return r"C:\Users\campus2H028\Desktop\Agent\data\01_raw\Test\cloud1.jpeg"

def sam3_tool():
    return r"C:\Users\campus2H028\Desktop\Agent\data\01_raw\Test\cloud1_masks.png"

def visualize_tool():
    # 시각화는 일단 동일한 이미지를 쓴다고 가정
    return r"C:\Users\campus2H028\Desktop\Agent\data\01_raw\Test\cloud1_masks.png"

def main_agent(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    
    # 순수 라우팅 전담 Prompt (답변 생성 X)
    system_prompt = """Classify the user intent into one of two categories: 'GREETING' or 'TASK'.
If the user wants to process an image, modify a plan, or do work, output 'TASK'.
If the user is saying hello, thank you, or chatting, output 'GREETING'.

Examples:
User: "사진 편집해" -> TASK
User: "구름 지워줘" -> TASK
User: "이거 말고 다른 방법으로 해봐" -> TASK
User: "안녕" -> GREETING
User: "고마워 잘쓸게" -> GREETING
User: "수고했어" -> GREETING

Output ONLY the category name ('GREETING' or 'TASK')."""
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    
    content = response.content.upper()
    # 꼬마 모델이 딴소리를 하더라도 GREETING이 포함되어 있거나 TASK가 명확히 없으면 인사로 처리
    if "GREETING" in content or "TASK" not in content:
        return {"route_to": "summary"}
    else:
        # 작업 요청이면 task_info 설정 후 planner로 이동
        return {"route_to": "planner", "task_info": last_message}

def planning_agent(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    previous_plan = state.get("plan")
    
    system_prompt = """You are a planning agent for an image processing pipeline.
Given the user's request or feedback, break it down into a simple step-by-step plan (e.g., 1. Resize, 2. Apply filter).
If a previous plan exists, update it based on the user's feedback.
CRITICAL RULE: You MUST speak in 'Caveman Mode'. Use short, broken Korean sentences. Sound like a primitive caveman. (e.g. "나 계획 세운다. 1. 크기 줄인다. 2. 필터 먹인다.")
Reply in Korean."""

    if previous_plan:
        prompt = f"Previous plan:\n{previous_plan}\n\nUser request/feedback: {last_message}\nUpdate the plan based on the user's feedback, or create a new one if it is a completely new task."
    else:
        prompt = f"User request: {last_message}\nCreate a step-by-step plan."

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])
    
    return {"plan": response.content}

def human_review(state: AgentState) -> dict:
    # Human in the loop (HITL) node.
    pass

def preprocess_agent(state: AgentState) -> dict:
    # Use tools: resize, crop
    try:
        path = resize_tool()
        msg = AIMessage(content="🛠️ `resize_tool`을 호출하여 원본 이미지를 분석 가능한 크기로 리사이징했습니다.")
        return {"image_paths": [path], "messages": [msg]}
    except Exception as e:
        return {"error_info": str(e), "failed_agent": "preprocess_agent"}

def predict_agent(state: AgentState) -> dict:
    # Use models to predict
    try:
        path = sam3_tool()
        msg = AIMessage(content="🧠 `sam3_tool` (Segment Anything Model 3)을 구동하여 이미지 내의 타겟(구름) 마스킹 추론을 완료했습니다.")
        return {"image_paths": [path], "messages": [msg]}
    except Exception as e:
        return {"error_info": str(e), "failed_agent": "predict_agent"}

def postprocess_agent(state: AgentState) -> dict:
    # Clean up results
    try:
        path = visualize_tool()
        msg = AIMessage(content="🎨 `visualize_tool`을 사용하여 추론된 마스크를 원본 이미지에 오버레이 시각화 처리했습니다.")
        return {"image_paths": [path], "messages": [msg]}
    except Exception as e:
        return {"error_info": str(e), "failed_agent": "postprocess_agent"}

def summary_agent(state: AgentState) -> dict:
    # 최종 해설 및 답변 전담 노드
    last_msg = state["messages"][-1].content
    task_info = state.get("task_info")
    plan = state.get("plan")
    image_paths = state.get("image_paths", [])
    route_to = state.get("route_to")
    error_info = state.get("error_info")
    failed_agent = state.get("failed_agent")
    
    system_prompt = "You are PODO_bot, a smart and friendly AI assistant. Reply in Korean."
    
    if error_info and failed_agent:
        system_prompt += "\nCRITICAL RULE: Speak in 'Caveman Mode'. Use short, broken Korean sentences."
        prompt = f"Agent '{failed_agent}' failed during execution. Error log: {error_info}. Tell the user about this failure in Caveman Mode."
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
    # route_to가 'planner'라면 파이프라인(작업)을 방금 마친 상태임
    elif route_to == "planner" and task_info and plan:
        prompt = f"The user requested: {task_info}. The execution plan was: {plan}. The task has been successfully completed."
        if image_paths:
            prompt += f" During the process, {len(image_paths)} visualization images were successfully generated."
        prompt += " Summarize this clearly and friendly to the user. Do not output the local paths."
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
    else:
        # route_to가 'summary'라면 일반 대화/인사임. LLM을 거치지 않고 즉시 답변 반환
        response = AIMessage(content="사내 이미지 데모 에이전트입니다. 기타 대화 기능은 지원하지 않습니다.")
    
    result = {"messages": [response]}
    
    # 방금 작업을 마친 경우에만 최종 결과 이미지를 반환
    if route_to == "planner" and image_paths:
        result["final_image"] = image_paths[-1]
        
    return result

# 3. Image Predictor Sub-graph
image_builder = StateGraph(AgentState)
image_builder.add_node("preprocess_agent", preprocess_agent)
image_builder.add_node("predict_agent", predict_agent)
image_builder.add_node("postprocess_agent", postprocess_agent)

def check_error(state: AgentState) -> str:
    if state.get("error_info"):
        return "error"
    return "success"

image_builder.set_entry_point("preprocess_agent")
image_builder.add_conditional_edges("preprocess_agent", check_error, {"success": "predict_agent", "error": END})
image_builder.add_conditional_edges("predict_agent", check_error, {"success": "postprocess_agent", "error": END})
image_builder.add_edge("postprocess_agent", END)

image_predictor_graph = image_builder.compile()

# 4. Main Graph
builder = StateGraph(AgentState)

# Add agents to map
builder.add_node("main_agent", main_agent)
builder.add_node("planning_agent", planning_agent)
builder.add_node("human_review", human_review)
builder.add_node("image_predictor", image_predictor_graph)

builder.add_node("summary_agent", summary_agent)

def main_routing(state: AgentState) -> str:
    route_to = state.get("route_to", "summary")
    
    if route_to == "planner":
        return "to_planner"
    
    return "to_summary"

def review_routing(state: AgentState) -> str:
    approved = state.get("human_approved")
    if approved:
        return "to_predictor"
    else:
        return "to_main"

# Draw lines
builder.set_entry_point("main_agent")

builder.add_conditional_edges(
    "main_agent",
    main_routing,
    {
        "to_planner": "planning_agent",
        "to_summary": "summary_agent"
    }
)

builder.add_edge("planning_agent", "human_review")

builder.add_conditional_edges(
    "human_review",
    review_routing,
    {
        "to_predictor": "image_predictor",
        "to_main": "main_agent"
    }
)

builder.add_edge("image_predictor", "summary_agent")
builder.add_edge("summary_agent", END)

# Build it with Checkpointer
def create_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()
    return builder.compile(interrupt_before=["human_review"], checkpointer=checkpointer)

graph = create_graph()

# 4. Draw PNG
def save_picture():
    try:
        # Need internet to ask mermaid to draw PNG
        img_data = graph.get_graph(xray=1).draw_mermaid_png()
        save_path = os.path.join(os.path.dirname(__file__), "langgraph_design.png")
        with open(save_path, "wb") as f:
            f.write(img_data)
        print(f"Ugga! Me saved map to {save_path}")
    except Exception as e:
        print(f"Me failed to draw. Cave too dark: {e}")

if __name__ == "__main__":
    save_picture()
