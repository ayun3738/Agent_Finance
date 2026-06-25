import operator
import os
from typing import Annotated, TypedDict, Sequence, Any, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
import chromadb
from sentence_transformers import SentenceTransformer

embed_model = None
def get_embed_model():
    global embed_model
    if embed_model is None:
        print("우가! 검색을 위해 CLIP 모델을 뇌에 쑤셔넣고 있다...")
        embed_model = SentenceTransformer('clip-ViT-B-32')
    return embed_model


# LLM 초기화 (Ollama 로컬 모델)
# num_predict=1000을 설정하여 모델이 최대 1000토큰까지만 생성하도록 강제 제한 (무한루프 방지)
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0, num_predict=1000)

# 1. State (Memory)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    task_info: str
    image_size: tuple
    plan: Optional[str]
    human_approved: Optional[bool]
    route_to: str  # 라우팅을 결정할 플래그 ("END", "planner" 등)
    image_paths: Annotated[list[str], operator.add]  # 누적할 이미지 경로 리스트
    image_analysis_result: Optional[str]  # VLM 분석 결과
    final_image: Optional[str]  # summary_agent가 반환할 최종 이미지 경로
    error_info: Optional[str]
    failed_agent: Optional[str]
    retrieved_images: Optional[list[str]] # RAG 검색으로 찾은 이미지들

# 임시 툴(Tool) 정의
def resize_tool():
    return r"C:\Users\campus2H028\Desktop\Agent\data\01_raw\Test\cloud1.jpeg"

def sam3_tool():
    return r"C:\Users\campus2H028\Desktop\Agent\data\01_raw\Test\cloud1_masks.png"

def visualize_tool():
    # 시각화는 일단 동일한 이미지를 쓴다고 가정
    return r"C:\Users\campus2H028\Desktop\Agent\data\01_raw\Test\cloud1_masks.png"

async def main_agent(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    
    # 순수 라우팅 전담 Prompt (답변 생성 X)
    system_prompt = """Classify the user intent into one of three categories: 'GREETING', 'TASK', or 'SEARCH'.
If the user wants to process an image, modify a plan, or do work, output 'TASK'.
If the user is asking to find or search for an image in the database (e.g. "구름 사진 있어?", "DB에서 강아지 찾아줘"), output 'SEARCH'.
If the user is saying hello, thank you, or chatting, output 'GREETING'.

Examples:
User: "사진 편집해" -> TASK
User: "구름 지워줘" -> TASK
User: "DB에 구름 사진 있냐?" -> SEARCH
User: "비슷한 사진 찾아봐" -> SEARCH
User: "안녕" -> GREETING

Output ONLY the category name ('GREETING', 'TASK', or 'SEARCH')."""
    
    response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    
    content = response.content.upper()
    
    if "SEARCH" in content:
        return {"route_to": "rag", "task_info": last_message}
    elif "GREETING" in content or ("TASK" not in content and "SEARCH" not in content):
        return {"route_to": "summary"}
    else:
        # 작업 요청이면 task_info 설정 후 planner로 이동
        return {"route_to": "planner", "task_info": last_message}

async def planning_agent(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    previous_plan = state.get("plan")
    
    system_prompt = """너는 이미지 처리 작업을 위한 원시인 플래너(계획자)다.
명령: 주인의 요청이나 거절 피드백을 받으면, 반드시 그에 맞춰서 단계별 행동 계획(1. 뭐한다, 2. 뭐한다)을 세워야 한다.
이전에 세운 계획이 있다면, 주인의 피드백을 '무조건' 반영해서 계획을 뜯어고쳐라!
CRITICAL RULE: 반드시 '원시인(Caveman)' 말투로 대답해라. 짧고 투박한 한국어를 써라 우가! (예: "나 계획 짠다. 1. 그림 자른다. 2. 색깔 바꾼다 우가!")"""

    if previous_plan:
        prompt = f"이전 계획:\n{previous_plan}\n\n주인의 피드백 (거절 사유): {last_message}\n\n명령: 주인이 니 계획을 거절했다! 주인의 피드백을 무조건 반영해서 기존 계획을 싹 다 뜯어고쳐라 우가!"
    else:
        prompt = f"주인의 요청: {last_message}\n\n명령: 주인의 요청을 해결하기 위한 단계별 행동 계획을 세워라 우가!"

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])
    
    return {"plan": response.content, "messages": [response]}

async def human_review(state: AgentState) -> dict:
    # Human in the loop (HITL) node.
    pass

async def image_rag_agent(state: AgentState) -> dict:
    last_message = state["messages"][-1].content
    try:
        model = get_embed_model()
        vector = model.encode(last_message).tolist()
        
        # ChromaDB 검색
        DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
        client = chromadb.PersistentClient(path=DB_DIR)
        collection = client.get_or_create_collection(name="image_rag_cosine")
        
        results = collection.query(
            query_embeddings=[vector],
            n_results=5
        )
        
        retrieved_paths = []
        distances = []
        if results and results['ids'] and len(results['ids'][0]) > 0:
            retrieved_paths = results['ids'][0]
            if 'distances' in results and results['distances']:
                distances = results['distances'][0]
                
        content_lines = [f"🔍 DB에서 {len(retrieved_paths)}개의 비슷한 이미지를 뒤져왔다 우가!"]
        for i in range(len(retrieved_paths)):
            dist = distances[i] if distances else 0.0
            # 코사인 거리(Cosine Distance)를 코사인 유사도(Cosine Similarity)로 변환
            sim = 1.0 - dist
            sim_rounded = round(sim, 4)
            content_lines.append(f"{sim_rounded} | {retrieved_paths[i]}")
            
        msg = AIMessage(content="\n".join(content_lines))
        return {"retrieved_images": retrieved_paths, "messages": [msg], "image_paths": retrieved_paths}
    except Exception as e:
        return {"error_info": str(e), "failed_agent": "image_rag_agent"}

async def image_analysis_agent(state: AgentState) -> dict:
    image_paths = state.get("image_paths", [])
    if not image_paths:
        return {}
    
    # 임시 VLM 모의 동작 (실제 VLM 연동 시 이곳에 구현)
    msg = AIMessage(content=f"👀 우가우가! 네가 올린 이미지({len(image_paths)}장) 쳐다봤다! 구름이 예쁘게 떠 있다 우가!")
    return {"image_analysis_result": msg.content, "messages": [msg]}

async def preprocess_agent(state: AgentState) -> dict:
    # Use tools: resize, crop
    try:
        path = resize_tool()
        msg = AIMessage(content="🛠️ `resize_tool`을 호출하여 원본 이미지를 분석 가능한 크기로 리사이징했습니다.")
        return {"image_paths": [path], "messages": [msg]}
    except Exception as e:
        return {"error_info": str(e), "failed_agent": "preprocess_agent"}

async def predict_agent(state: AgentState) -> dict:
    # Use models to predict
    try:
        path = sam3_tool()
        msg = AIMessage(content="🧠 `sam3_tool` (Segment Anything Model 3)을 구동하여 이미지 내의 타겟(구름) 마스킹 추론을 완료했습니다.")
        return {"image_paths": [path], "messages": [msg]}
    except Exception as e:
        return {"error_info": str(e), "failed_agent": "predict_agent"}

async def postprocess_agent(state: AgentState) -> dict:
    # Clean up results
    try:
        path = visualize_tool()
        msg = AIMessage(content="🎨 `visualize_tool`을 사용하여 추론된 마스크를 원본 이미지에 오버레이 시각화 처리했습니다.")
        return {"image_paths": [path], "messages": [msg]}
    except Exception as e:
        return {"error_info": str(e), "failed_agent": "postprocess_agent"}

async def summary_agent(state: AgentState) -> dict:
    # 최종 해설 및 답변 전담 노드
    last_msg = state["messages"][-1].content
    task_info = state.get("task_info")
    plan = state.get("plan")
    image_paths = state.get("image_paths", [])
    route_to = state.get("route_to")
    error_info = state.get("error_info")
    failed_agent = state.get("failed_agent")
    image_analysis_result = state.get("image_analysis_result")
    retrieved_images = state.get("retrieved_images")
    
    system_prompt = "You are PODO_bot, a smart and friendly AI assistant. Reply in Korean."
    
    if error_info and failed_agent:
        system_prompt += "\nCRITICAL RULE: Speak in 'Caveman Mode'. Use short, broken Korean sentences."
        prompt = f"Agent '{failed_agent}' failed during execution. Error log: {error_info}. Tell the user about this failure in Caveman Mode."
        response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
    # route_to가 'planner'라면 파이프라인(작업)을 방금 마친 상태임
    elif route_to == "planner" and task_info and plan:
        prompt = f"The user requested: {task_info}. The execution plan was: {plan}. The task has been successfully completed."
        if image_analysis_result:
            prompt += f"\nImage Analysis Result: {image_analysis_result}"
        if image_paths:
            prompt += f" During the process, {len(image_paths)} visualization images were successfully generated."
        prompt += " Summarize this clearly and friendly to the user. Do not output the local paths."
        response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
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

async def check_error(state: AgentState) -> str:
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
builder.add_node("image_analysis_agent", image_analysis_agent)
builder.add_node("image_rag_agent", image_rag_agent)
builder.add_node("human_review", human_review)
builder.add_node("image_predictor", image_predictor_graph)

builder.add_node("summary_agent", summary_agent)

async def main_routing(state: AgentState) -> list[str]:
    route_to = state.get("route_to", "summary")
    
    if route_to == "planner":
        routes = ["planning_agent"]
        if state.get("image_paths"):
            routes.append("image_analysis_agent")
        return routes
    elif route_to == "rag":
        return ["image_rag_agent"]
    
    return ["summary_agent"]

async def review_routing(state: AgentState) -> str:
    approved = state.get("human_approved")
    if approved:
        return "image_predictor"
    else:
        # 강제 취소인 경우 main_agent로 보내서 대답 후 종료하게 만듦
        messages = state.get("messages", [])
        if messages and "[시스템] 사용자가 이 계획을 강제로 취소" in getattr(messages[-1], "content", ""):
            return "main_agent"
        # 일반 거절인 경우 불필요한 라우팅을 거치지 않고 바로 planning_agent로 직행하여 재계획
        return "planning_agent"

# Draw lines
builder.set_entry_point("main_agent")

builder.add_conditional_edges(
    "main_agent",
    main_routing,
    ["planning_agent", "image_analysis_agent", "image_rag_agent", "summary_agent"]
)

builder.add_edge("planning_agent", "human_review")
builder.add_edge("image_analysis_agent", "human_review")
builder.add_edge("image_rag_agent", END)

builder.add_conditional_edges(
    "human_review",
    review_routing,
    {
        "image_predictor": "image_predictor",
        "planning_agent": "planning_agent",
        "main_agent": "main_agent"
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
