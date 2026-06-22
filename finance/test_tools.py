from finance.tools import (
    AgentState,
    get_kospi_tickers,
    fetch_dart_disclosures,
    fetch_stock_prices,
    fetch_news,
    analyze_with_llm
)

def test_tools():
    print("--- 테스트 시작 ---")
    
    state: AgentState = {
        "tickers": [],
        "disclosures": {},
        "stock_prices": {},
        "news_articles": {},
        "analysis_report": ""
    }
    
    print("1. get_kospi_tickers 테스트")
    res1 = get_kospi_tickers(state)
    print(res1)
    state.update(res1)
    
    print("\n2. fetch_dart_disclosures 테스트")
    res2 = fetch_dart_disclosures(state)
    print(res2)
    state.update(res2)
    
    print("\n3. fetch_stock_prices 테스트")
    res3 = fetch_stock_prices(state)
    print(res3)
    state.update(res3)
    
    print("\n4. fetch_news 테스트")
    res4 = fetch_news(state)
    print(res4)
    state.update(res4)
    
    print("\n5. analyze_with_llm 테스트")
    res5 = analyze_with_llm(state)
    print(res5)
    
    print("\n--- 테스트 종료 ---")

if __name__ == "__main__":
    test_tools()
