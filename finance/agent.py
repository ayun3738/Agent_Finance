from langgraph.graph import StateGraph, START, END
import schedule
import time
from finance.tools import (
    AgentState,
    get_kospi_tickers,
    fetch_dart_disclosures,
    fetch_stock_prices,
    fetch_news,
    analyze_with_llm
)
from finance.awakeplus_agent import awakeplus_app

def call_awakeplus(state: AgentState):
    print("\n[Main Agent] AwakePlus 서브 에이전트 호출 중...")
    res = awakeplus_app.invoke({
        "scraped_disclosures": [],
        "filtered_disclosures": [],
        "final_disclosures": []
    })
    
    final_data = res.get("final_disclosures", [])
    print(f"[Main Agent] AwakePlus에서 총 {len(final_data)}건의 공시 수집 완료.")
    
    # 상태 병합 로직 (필요시 추가)
    return {"disclosures": state.get("disclosures", {})}

# 3. 랭그래프(LangGraph) 워크플로우 구성
workflow = StateGraph(AgentState)
workflow.add_node("call_awakeplus", call_awakeplus)
workflow.add_node("get_tickers", get_kospi_tickers)
workflow.add_node("fetch_dart", fetch_dart_disclosures)
workflow.add_node("fetch_stock", fetch_stock_prices)
workflow.add_node("fetch_news", fetch_news)
workflow.add_node("analyze", analyze_with_llm)

# 엣지 연결 (병렬 처리 지원 구조)
workflow.add_edge(START, "call_awakeplus")
workflow.add_edge("call_awakeplus", "get_tickers")
workflow.add_edge("get_tickers", "fetch_dart")
workflow.add_edge("get_tickers", "fetch_stock")
workflow.add_edge("get_tickers", "fetch_news")
workflow.add_edge(["fetch_dart", "fetch_stock", "fetch_news"], "analyze")
workflow.add_edge("analyze", END)

app = workflow.compile()

# 4. 스케줄러 등록 (매일 오전 8시)
def daily_job():
    print("아침 8시! KOSPI 분석 에이전트를 가동합니다.")
    result = app.invoke({"tickers": [], "disclosures": {}, "stock_prices": {}, "news_articles": {}, "analysis_report": ""})
    print(result["analysis_report"])

schedule.every().day.at("08:00").do(daily_job)

if __name__ == "__main__":
    print("스케줄러 대기 중...")
    while True:
        schedule.run_pending()
        time.sleep(60)
