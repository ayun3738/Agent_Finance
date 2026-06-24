import os
import uuid
import chainlit as cl
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage
import sys
import asyncio
import threading
import queue

from chainlit.input_widget import Switch

# Agent 루트 디렉토리를 path에 추가하여 PODO_bot 패키지 인식 문제 해결
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PODO_bot.langgraph_design import create_graph

load_dotenv()

# Global connection pool 대신 매번 커넥션을 여는 from_conn_string 사용 (트랜잭션 충돌 방지)
POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/postgres")

# 앱 구동 시 테이블 생성
db_connected = False
def setup_db():
    global db_connected
    try:
        with PostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
            checkpointer.setup()
        db_connected = True
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        db_connected = False

# 백그라운드에서 셋업 (join으로 메인 스레드 블로킹 방지)
setup_thread = threading.Thread(target=setup_db, daemon=True)
setup_thread.start()

@cl.on_chat_start
async def on_chat_start():
    # 설정 토글 UI 렌더링
    settings = await cl.ChatSettings(
        [
            Switch(id="ai_detail", label="AI Detail Logs (중간 과정 렌더링)", initial=False),
            Switch(id="auto_proceed", label="Auto Proceed (자동 승인 진행)", initial=False)
        ]
    ).send()
    cl.user_session.set("ai_detail", False)
    cl.user_session.set("auto_proceed", False)

    # 1. 페이지 새로고침(신규 접속) 마다 고유 thread_id 발급
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    
    if not db_connected:
        await cl.Message(content="🚨 **Postgres DB 연결에 실패했습니다.**\n로컬에 Postgres 서버가 떠있는지, `.env`의 `POSTGRES_URI`가 맞는지 확인해주세요!").send()
        return

    # 2. 발급된 thread_id를 화면에 표시
    await cl.Message(
        content=f"우가! 새로운 세션 시작했다.\n🔑 **당신의 Thread ID:** `{thread_id}`\n\n무엇을 도와줄까?"
    ).send()

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("ai_detail", settings["ai_detail"])
    cl.user_session.set("auto_proceed", settings["auto_proceed"])

def run_graph_sync(inputs, config, user_input, auto_proceed, q, stop_event):
    try:
        with PostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
            graph = create_graph(checkpointer=checkpointer)
            
            # 현재 상태 확인
            state = graph.get_state(config)
            
            if state.next and "human_review" in state.next:
                if user_input in ['y', 'yes', '승인']:
                    graph.update_state(config, {"human_approved": True}, as_node="human_review")
                    q.put(("msg", "✅ 주인이 승인했다! 예측기로 넘어간다!"))
                else:
                    # 거절 시 피드백 메시지를 state에 주입하여 LLM이 참고할 수 있도록 함
                    reason = user_input.replace("reject:", "") if user_input.startswith("reject:") else user_input
                    graph.update_state(config, {
                        "human_approved": False,
                        "messages": [HumanMessage(content=f"계획이 거절되었습니다. 사용자 피드백: {reason}")]
                    }, as_node="human_review")
                    q.put(("msg", f"❌ 계획이 거절되었습니다. (사유: {reason}) 다시 플래너와 대화합니다."))
                inputs = None # 재개 시 기존 inputs 비우기
                
            while True:
                # 스트리밍 실행 (일반 실행 또는 재개)
                for namespace, event in graph.stream(inputs, config, subgraphs=True):
                    if stop_event.is_set():
                        q.put(("msg", "🛑 중지됨!"))
                        break
                    q.put(("event", event))
                    
                if stop_event.is_set():
                    break
                    
                # 스트림이 끝나고 난 후 인터럽트 상태 확인
                state = graph.get_state(config)
                if state.next and "human_review" in state.next:
                    if auto_proceed:
                        graph.update_state(config, {"human_approved": True}, as_node="human_review")
                        q.put(("msg", "⚡ **[Auto Proceed] 켜짐:** 계획을 자동으로 승인하고 예측 파이프라인을 가동합니다!"))
                        inputs = None
                        continue  # 다시 루프 돌아서 자동 스트림 진행
                    else:
                        q.put(("ask_approval", None))
                        break
                else:
                    # 그래프가 완전히 끝남
                    break
                
    except Exception as e:
        q.put(("error", str(e)))
    finally:
        q.put(("done", None))

@cl.on_message
async def on_message(message: cl.Message):
    if not db_connected:
        await cl.Message(content="DB가 없어서 아무것도 못한다 우가!").send()
        return

    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=message.content)]}
    user_input = message.content.strip().lower()
    auto_proceed = cl.user_session.get("auto_proceed", False)
    
    q = queue.Queue()
    stop_event = threading.Event()
    t = threading.Thread(target=run_graph_sync, args=(inputs, config, user_input, auto_proceed, q, stop_event))
    t.start()
    
    try:
        while True:
            while not q.empty():
                msg_type, data = q.get()
                if msg_type == "done":
                    return
                elif msg_type == "error":
                    await cl.Message(content=f"🚨 에러 발생: {data}").send()
                    return
                elif msg_type == "msg":
                    await cl.Message(content=data).send()
                elif msg_type == "ask_approval":
                    # 인터랙티브 버튼 띄우기
                    res = await cl.AskActionMessage(
                        content="⚠️ **[인간 승인 필요]**\n계획 에이전트가 위와 같이 작업 계획을 세웠습니다. 승인하시겠습니까?",
                        actions=[
                            cl.Action(name="approve", payload={"value": "approve"}, label="✅ 승인 (계속 진행)"),
                            cl.Action(name="reject", payload={"value": "reject"}, label="❌ 거절 (피드백 입력)")
                        ]
                    ).send()
                    
                    if res and res.get("payload", {}).get("value") == "approve":
                        await on_message(cl.Message(content="y"))
                        return
                    elif res and res.get("payload", {}).get("value") == "reject":
                        reason_res = await cl.AskUserMessage(content="❌ 승인이 거절되었습니다. 수정 사항이나 거절 사유를 입력해주세요.", timeout=600).send()
                        if reason_res:
                            reason = reason_res["output"]
                            await on_message(cl.Message(content=f"reject:{reason}"))
                        else:
                            await on_message(cl.Message(content="reject:사유 없음"))
                        return
                        
                elif msg_type == "event":
                    ai_detail = cl.user_session.get("ai_detail", False)
                    for key, value in data.items():
                        # image_predictor(서브그래프 래퍼 노드)는 전체 출력을 중복으로 뱉으므로 무조건 무시
                        if key == "image_predictor":
                            continue
                            
                        # AI Detail 토글이 꺼져있을 때 중간 과정 노드의 출력 숨기기
                        if not ai_detail and key in ["preprocess_agent", "predict_agent", "postprocess_agent"]:
                            continue
                            
                        # AI Detail 토글이 켜져 있으면, 현재 실행을 마친 노드의 이름을 진행 과정으로 표시
                        if ai_detail:
                            await cl.Message(content=f"🔄 `{key}` 호출 및 동작 완료").send()
                            
                        elements = []
                        # 새로 추가된 이미지(로컬 파일)가 있다면 cl.Image 객체 생성
                        if "image_paths" in value and value["image_paths"]:
                            latest_img_path = value["image_paths"][0] # 이번 노드에서 생성한 이미지 (스트림 단위)
                            if os.path.exists(latest_img_path):
                                img = cl.Image(path=latest_img_path, name=f"{key}_output", display="inline")
                                elements.append(img)
                                
                        # 최종 요약(summary_agent)일 경우, 에이전트가 직접 반환한 final_image를 꺼내서 첨부
                        if "final_image" in value and value["final_image"]:
                            final_img_path = value["final_image"]
                            if os.path.exists(final_img_path):
                                img = cl.Image(path=final_img_path, name="final_visualized_output", display="inline")
                                elements.append(img)
                                
                        # 노드가 반환한 데이터 중 메시지가 있다면 출력
                        if "messages" in value and value["messages"]:
                            last_msg = value["messages"][-1]
                            if hasattr(last_msg, "content"):
                                await cl.Message(author=key, content=last_msg.content, elements=elements).send()
                            else:
                                await cl.Message(content=f"⚙️ `[{key}]` 작동 완료!", elements=elements).send()
                        # 계획(plan)이 생성되었다면 출력
                        elif "plan" in value and value["plan"]:
                            await cl.Message(author=key, content=f"📋 **계획 수립됨:**\n{value['plan']}", elements=elements).send()
                        else:
                            # 텍스트는 없고 이미지만 있는 경우 (preprocess, predict 등)
                            if elements:
                                await cl.Message(author=key, content=f"🖼️ `[{key}]` 시각화 결과:", elements=elements).send()
                            else:
                                # 단순 라우팅 노드 등은 출력하지 않고 무시
                                pass
            
            if not t.is_alive() and q.empty():
                break
                
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        # Chainlit Stop 버튼을 누르면 이 예외가 발생함
        stop_event.set()
        await cl.Message(content="🛑 **사용자에 의해 작업이 강제 중지되었습니다.**").send()
        raise
