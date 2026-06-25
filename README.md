finance

PODO_bot

```uv
uv run chainlit run PODO_bot/chainlit_app.py | Tee-Object -FilePath "server.log"
```



기본 동작

- 비동기 ollama 로컬 구동 ()
- thread_id 기반 채팅 DB 관리
- python tool들 연결(gitea repository 기반의)
- 작업 요청 -> 계획 수립 -> 승인 절차 -> 작업 수행 -> 결과 보고
- 



1. UI 기능
   1. 채팅 기능 : 기본 채팅 + 승인 거절 시, 메시지 전송
   2. setting : 진행 과정 detail 표시, 승인 자동화 옵션
2. 이미지 첨부 기능
