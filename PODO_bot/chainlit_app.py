import os
import uuid
import chainlit as cl
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from chainlit.input_widget import Switch, TextInput

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
async def ensure_db_setup():
    global db_connected
    if db_connected:
        return
    print("우가! DB 셋업 시도한다!")
    try:
        async with AsyncPostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
            await checkpointer.setup()
        db_connected = True
        print("우가! DB 연결 성공했다!")
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        db_connected = False

# --- 트래픽 관리를 위한 전역 큐(티켓) 변수 ---
current_ticket = 0
serving_ticket = 0
ticket_condition = asyncio.Condition()
cancelled_threads = set()
# -----------------------------------------------

async def clear_approval_actions():
    actions = cl.user_session.get("approval_actions", [])
    if actions:
        for a in actions:
            try:
                await a.remove()
            except Exception:
                pass
        cl.user_session.set("approval_actions", [])

@cl.on_chat_start
async def on_chat_start():
    await ensure_db_setup()

    # 1. 페이지 새로고침(신규 접속) 마다 고유 thread_id 발급
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)

    # 설정 토글 UI 렌더링 (Thread ID 복사용 필드 추가)
    settings = await cl.ChatSettings(
        [
            Switch(id="ai_detail", label="AI Detail Logs (중간 과정 렌더링)", initial=False),
            Switch(id="auto_proceed", label="Auto Proceed (자동 승인 진행)", initial=False),
            TextInput(id="display_thread_id", label="🔑 Thread ID (복사용)", initial=thread_id)
        ]
    ).send()
    cl.user_session.set("ai_detail", False)
    cl.user_session.set("auto_proceed", False)
    
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

# 분리된 LangGraph 실제 처리 로직
async def process_graph_message(message: cl.Message, user_input: str, thread_id: str, auto_proceed: bool, ai_detail: bool):
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        async with AsyncPostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
            graph = create_graph(checkpointer=checkpointer)
            
            state_before = await graph.aget_state(config)
            before_message_ids = {m.id for m in state_before.values.get("messages", []) if getattr(m, "id", None)}
            
            state = state_before
            
            inputs = None
            if state.next and "human_review" in state.next:
                # 사용자가 텍스트로 응답했든 버튼을 눌렀든, 이전 턴에 띄워둔 버튼들은 모두 지운다.
                await clear_approval_actions()
                
                user_input_lower = user_input.lower()
                if user_input_lower == 'abort':
                    await graph.aupdate_state(config, {
                        "human_approved": False,
                        "messages": [HumanMessage(content="[시스템] 사용자가 이 계획을 강제로 취소했습니다. 즉시 사용자에게 '취소했다 우가!' 라고 대답하고 다른 작업은 일절 하지 마세요.")]
                    }, as_node="human_review")
                    inputs = None
                elif user_input_lower in ['y', 'yes', '승인', 'approve']:
                    await graph.aupdate_state(config, {"human_approved": True}, as_node="human_review")
                    await cl.Message(content="✅ 주인이 승인했다! 예측기로 넘어간다!").send()
                else:
                    reason = user_input.replace("reject:", "") if user_input.startswith("reject:") else user_input
                    await graph.aupdate_state(config, {
                        "human_approved": False,
                        "messages": [HumanMessage(content=f"계획이 거절되었습니다. 사용자 피드백: {reason}")]
                    }, as_node="human_review")
                    await cl.Message(content=f"❌ 계획이 거절되었습니다. (사유: {reason}) 다시 플래너와 대화합니다.").send()
            else:
                user_image_paths = []
                if message.elements:
                    for el in message.elements:
                        if hasattr(el, "mime") and el.mime and "image" in el.mime:
                            user_image_paths.append(el.path)
                            
                inputs = {
                    "messages": [HumanMessage(content=user_input)],
                    "image_paths": user_image_paths
                }
                
            try:
                while True:
                    # 스트리밍 실행 (일반 실행 또는 재개)
                    async for namespace, event in graph.astream(inputs, config, subgraphs=True):
                        for key, value in event.items():
                            if key == "image_predictor":
                                continue
                            
                            if not ai_detail and key in ["preprocess_agent", "predict_agent", "postprocess_agent"]:
                                continue
                            
                            if ai_detail:
                                await cl.Message(content=f"🔄 `{key}` 호출 및 동작 완료").send()
                            
                            elements = []
                            if "image_paths" in value and value["image_paths"]:
                                latest_img_path = value["image_paths"][0] # 이번 노드에서 생성한 이미지
                                if os.path.exists(latest_img_path):
                                    img = cl.Image(path=latest_img_path, name=f"{key}_output", display="inline")
                                    elements.append(img)
                                
                            if "final_image" in value and value["final_image"]:
                                final_img_path = value["final_image"]
                                if os.path.exists(final_img_path):
                                    img = cl.Image(path=final_img_path, name="final_visualized_output", display="inline")
                                    elements.append(img)
                                
                            if "messages" in value and value["messages"]:
                                last_msg = value["messages"][-1]
                                if hasattr(last_msg, "content"):
                                    await cl.Message(author=key, content=last_msg.content, elements=elements).send()
                                else:
                                    await cl.Message(content=f"⚙️ `[{key}]` 작동 완료!", elements=elements).send()
                            elif "plan" in value and value["plan"]:
                                await cl.Message(author=key, content=f"📋 **계획 수립됨:**\n{value['plan']}", elements=elements).send()
                            else:
                                if elements:
                                    await cl.Message(author=key, content=f"🖼️ `[{key}]` 시각화 결과:", elements=elements).send()

                    # 스트림 종료 후 상태 확인
                    state = await graph.aget_state(config)
                    if state.next and "human_review" in state.next:
                        if auto_proceed:
                            await graph.aupdate_state(config, {"human_approved": True}, as_node="human_review")
                            await cl.Message(content="⚡ **[Auto Proceed] 켜짐:** 계획을 자동으로 승인하고 예측 파이프라인을 가동합니다!").send()
                            inputs = None
                            continue
                        else:
                            actions=[
                                cl.Action(name="approve_action", payload={"value": "approve"}, label="✅ 승인 (계속 진행)"),
                                cl.Action(name="reject_action", payload={"value": "reject"}, label="❌ 거절 (채팅으로 사유 작성)"),
                                cl.Action(name="abort_action", payload={"value": "abort"}, label="🛑 작업 취소")
                            ]
                            cl.user_session.set("approval_actions", actions)
                            approval_msg = await cl.Message(
                                content="⚠️ **[인간 승인 필요]**\n계획 에이전트가 작업 계획을 세웠습니다. 승인하시겠습니까?\n거절하려면 버튼을 누르고 채팅창에 `거절: 이유`를 입력해라 우가!",
                                actions=actions
                            ).send()
                            cl.user_session.set("approval_msg_id", approval_msg.id)
                            break
                    else:
                        # 그래프 완전히 끝남
                        break

            except asyncio.CancelledError:
                # 작업 강제 취소 시, Time Travel을 사용하여 완벽하게 작업 시작 전(state_before)으로 롤백!
                # state_before의 config(체크포인트 ID)를 이어받아 새로운 체크포인트를 만듦으로써 중간의 모든 쓰레기를 무효화
                await graph.aupdate_state(state_before.config, {"error_info": "rollback"})
                
                await cl.Message(content="🛑 **사용자에 의해 작업이 강제 중지되었습니다. (질문 전 상태로 완벽 롤백 완료)**").send()
                
                # 만약 롤백된 상태가 승인을 기다리는 상태라면, 버튼을 다시 부활시킴!
                if state_before.next and "human_review" in state_before.next:
                    actions=[
                        cl.Action(name="approve_action", payload={"value": "approve"}, label="✅ 승인 (계속 진행)"),
                        cl.Action(name="reject_action", payload={"value": "reject"}, label="❌ 거절 (채팅으로 사유 작성)"),
                        cl.Action(name="abort_action", payload={"value": "abort"}, label="🛑 작업 취소")
                    ]
                    cl.user_session.set("approval_actions", actions)
                    
                    prev_plan = state_before.values.get("plan", "")
                    content_msg = "⚠️ **[인간 승인 필요]**\n작업이 중지되어 이전 계획 상태로 되돌아왔다 우가!"
                    if prev_plan:
                        content_msg += f"\n\n**[복구된 이전 계획]**\n{prev_plan}\n\n다시 선택해라 우가!"
                    else:
                        content_msg += "\n다시 선택해라 우가!"

                    approval_msg = await cl.Message(
                        content=content_msg,
                        actions=actions
                    ).send()
                    cl.user_session.set("approval_msg_id", approval_msg.id)
                raise

    except Exception as e:
        await cl.Message(content=f"🚨 에러 발생: {str(e)}").send()

@cl.action_callback("approve_action")
async def on_approve(action: cl.Action):
    await clear_approval_actions()
    msg_id = cl.user_session.get("approval_msg_id")
    if msg_id:
        msg = cl.Message(id=msg_id, content="✅ **[승인 완료]** 작업을 재개한다 우가!", actions=[])
        await msg.update()
        
    # 승인 메시지를 큐 핸들러에 전달하여 실행 (이 콜백이 끝날 때까지 UI 중지 버튼 활성화)
    msg = cl.Message(content="y")
    await queue_message_handler(msg)

@cl.action_callback("reject_action")
async def on_reject(action: cl.Action):
    await clear_approval_actions()
    msg_id = cl.user_session.get("approval_msg_id")
    if msg_id:
        msg = cl.Message(id=msg_id, content="❌ **[거절 대기]** 사유를 듣기 위해 귀를 열었다 우가!", actions=[])
        await msg.update()

    # AskUserMessage가 체인릿의 UI를 잠가버리는 버그를 유발하므로 일반 메시지로 안내하고 기존 채팅창을 그대로 활용
    await cl.Message(
        content="👉 봇이 멍청하게 계획을 짰다 우가!\n**무엇을 고칠지 거절 사유를 아래 '일반 채팅창'에 바로 적어라! (아무 말이나 치면 전부 거절 사유로 봇에게 전달된다)**"
    ).send()

@cl.action_callback("abort_action")
async def on_abort(action: cl.Action):
    await clear_approval_actions()
    msg_id = cl.user_session.get("approval_msg_id")
    if msg_id:
        msg = cl.Message(id=msg_id, content="🛑 **[작업 취소]** 사용자가 이 작업을 강제 종료했다 우가!", actions=[])
        await msg.update()
        
    # 취소 메시지를 큐에 던져서 CancelledError를 강제로 발생시킴
    msg = cl.Message(content="abort")
    await queue_message_handler(msg)

async def queue_message_handler(message: cl.Message):
    """메시지를 큐에 넣고 순서가 되면 process_graph_message를 호출하는 핸들러"""
    global current_ticket, serving_ticket
    
    await ensure_db_setup()
    
    if not db_connected:
        await cl.Message(content="DB가 없어서 아무것도 못한다 우가!").send()
        return

    thread_id = cl.user_session.get("thread_id")
    user_input = message.content.strip()
    auto_proceed = cl.user_session.get("auto_proceed", False)
    ai_detail = cl.user_session.get("ai_detail", False)

    # 티켓 발급
    async with ticket_condition:
        my_ticket = current_ticket
        current_ticket += 1

    queue_msg = None
    if my_ticket > serving_ticket:
        position = my_ticket - serving_ticket
        queue_msg = await cl.Message(content=f"⏳ 대기열에 추가되었다! 현재 앞에 **{position}개**의 작업이 대기 중이다 우가!").send()
        
        async with ticket_condition:
            while my_ticket > serving_ticket:
                await ticket_condition.wait()
                
                # 기다리는 중에 사용자가 이전 작업을 중지(Stop)해서 취소 플래그가 켜졌다면 같이 런(Abort)
                if thread_id in cancelled_threads:
                    if queue_msg:
                        queue_msg.content = "🛑 앞선 작업이 중지되어 대기열에 있던 이 요청도 같이 무효화(Abort) 되었다 우가!"
                        await queue_msg.update()
                    serving_ticket += 1
                    ticket_condition.notify_all()
                    return
                
                # 내 순서가 아니면 대기열 숫자 업데이트
                new_position = my_ticket - serving_ticket
                if new_position > 0 and queue_msg:
                    queue_msg.content = f"⏳ 순서가 당겨졌다! 현재 앞에 **{new_position}개**의 작업이 대기 중이다 우가!"
                    await queue_msg.update()
        
        # 내 순서가 됨
        if queue_msg:
            queue_msg.content = "🚀 내 차례다! 작업을 시작한다 우가!"
            await queue_msg.update()

    # 작업을 무사히 시작할 자격을 얻었으므로(혹은 대기열 없이 바로 실행되므로), 이전 취소 플래그가 남아있다면 해제
    if thread_id in cancelled_threads:
        cancelled_threads.remove(thread_id)

    # 순서가 되면 로직 처리
    try:
        await process_graph_message(message, user_input, thread_id, auto_proceed, ai_detail)
    except asyncio.CancelledError:
        # 실행 도중 사용자가 강제 중지 버튼을 눌렀다면, 뒤에 대기 중인 작업들도 연쇄적으로 무효화하도록 플래그 켜기
        cancelled_threads.add(thread_id)
        raise
    finally:
        # 처리 완료 후 다음 사람 호출
        async with ticket_condition:
            serving_ticket += 1
            ticket_condition.notify_all()

@cl.on_message
async def on_message(message: cl.Message):
    # 진입점에서는 단순히 queue 핸들러를 호출
    await queue_message_handler(message)
