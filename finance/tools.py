from typing import TypedDict, List, Dict
from langchain_ollama import ChatOllama

# 상태(State) 정의
class AgentState(TypedDict):
    tickers: List[str]
    disclosures: Dict[str, str]
    stock_prices: Dict[str, str]
    news_articles: Dict[str, str]
    analysis_report: str

def get_kospi_tickers(state: AgentState):
    # fdr을 통해 KOSPI 목록 수집 로직 (코스피 전체는 800개가 넘어 LLM 과부하 우려로 우선 일부만 테스트 권장)
    return {"tickers": ["삼성전자", "SK하이닉스"]}

def fetch_dart_disclosures(state: AgentState):
    # OpenDartReader를 통한 최근 공시 조회
    return {"disclosures": {"삼성전자": "최근 공시 없음", "SK하이닉스": "유상증자 결정..."}}

def fetch_stock_prices(state: AgentState):
    # FinanceDataReader로 현재 주가 조회
    return {"stock_prices": {"삼성전자": "75,000원", "SK하이닉스": "150,000원"}}

def fetch_news(state: AgentState):
    # Tavily API를 통한 기업명 포함 최근 기사 조회
    return {"news_articles": {"삼성전자": "반도체 호황 기대감 상승...", "SK하이닉스": "AI 메모리 수요 폭발..."}}

def analyze_with_llm(state: AgentState):
    # 로컬 LLM (Ollama - qwen2.5:1.5b 초경량) 호출하여 종합 데이터로 호재/악재 분석
    llm = ChatOllama(model="qwen2.5:1.5b")
    # Prompt 작성 로직 생략... (공시, 주가, 뉴스 데이터를 프롬프트에 주입)
    report = "### 삼성전자 분석 결과: 호재 예상..."
    return {"analysis_report": report}
