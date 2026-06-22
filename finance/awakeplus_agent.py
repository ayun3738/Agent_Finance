import httpx
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import FinanceDataReader as fdr
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
import os
from opendartreader import OpenDartReader
from dotenv import load_dotenv

load_dotenv()
dart_api_key = os.environ.get("DART_API_KEY")
dart = OpenDartReader(dart_api_key) if dart_api_key else None

class AwakePlusState(TypedDict):
    scraped_disclosures: List[Dict[str, Any]]
    filtered_disclosures: List[Dict[str, Any]]
    final_disclosures: List[Dict[str, Any]]

def scrape_awakeplus(state: AwakePlusState):
    print("\n[AwakePlus] 1. 최근 7일치 공시 스크래핑 시작...")
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    page = 1
    scraped_data = []
    stop_scraping = False
    
    with httpx.Client() as client:
        while not stop_scraping:
            url = f"https://www.awakeplus.co.kr/data?page={page}"
            res = client.get(url)
            if res.status_code != 200:
                print(f"  - 페이지 {page} 접속 실패. 종료.")
                break
                
            soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select('.custom-table tbody tr')
            
            if not rows:
                break
                
            for row in rows:
                title_elem = row.select_one('.cell-title a')
                name_elem = row.select_one('.cell-stock-name')
                code_elem = row.select_one('.cell-code span')
                date_elem = row.select_one('.cell-date')
                
                if not (title_elem and name_elem and code_elem and date_elem):
                    continue
                    
                title = title_elem.text.strip()
                link = "https://www.awakeplus.co.kr" + title_elem['href']
                name = name_elem.text.strip()
                code = code_elem.text.strip()
                date_str = date_elem.text.strip()
                
                try:
                    # AwakePlus 날짜 형식: 2026-06-22 11:38:21
                    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    continue
                
                # 7일 전보다 오래된 데이터면 그만 긁음
                if dt < seven_days_ago:
                    stop_scraping = True
                    break
                    
                scraped_data.append({
                    "title": title,
                    "name": name,
                    "code": code,
                    "date": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "link": link
                })
            
            print(f"  - 페이지 {page} 파싱 완료. (누적: {len(scraped_data)}건)")
            page += 1
            if page > 50: # 무한루프 방지
                break
                
    return {"scraped_disclosures": scraped_data}

def filter_new_disclosures(state: AwakePlusState):
    print("\n[AwakePlus] 2. KOSPI 및 메타데이터 기준으로 필터링...")
    scraped = state.get("scraped_disclosures", [])
    if not scraped:
        return {"filtered_disclosures": []}
        
    kospi_list = fdr.StockListing('KOSPI')
    kospi_codes = set(kospi_list['Code'].tolist())
    
    filtered = []
    for item in scraped:
        code = item['code']
        # KOSPI 종목만 가져오기
        if code in kospi_codes:
            filtered.append(item)
            
    # TODO: meta.xlsx 데이터 읽어서 이미 처리된 공시는 제외하는 로직 추가 가능
    print(f"  - 총 {len(scraped)}건 중 KOSPI 종목 {len(filtered)}건 필터링 완료.")
    return {"filtered_disclosures": filtered}

def fetch_dart_details(state: AwakePlusState):
    print("\n[AwakePlus] 3. 필터링된 KOSPI 종목 Dart 공시 정보 가져오기...")
    filtered = state.get("filtered_disclosures", [])
    
    final_data = []
    if not dart:
        print("  - DART_API_KEY가 없어 Dart 원문을 가져올 수 없습니다.")
        return {"final_disclosures": filtered}
        
    for item in filtered:
        code = item['code']
        try:
            dart_res = dart.list(code)
            if dart_res is not None and not dart_res.empty:
                # 가장 최근 공시 1개만 매칭 (정확히는 제목, 날짜로 매칭해야 하지만 임시로 최신 1개)
                item['dart_rcept_dt'] = dart_res['rcept_dt'].iloc[0]
                item['dart_report_nm'] = dart_res['report_nm'].iloc[0]
        except Exception as e:
            pass # 못 가져와도 패스
            
        final_data.append(item)
        
    print(f"  - Dart 세부 정보 수집 완료.")
    return {"final_disclosures": final_data}

# LangGraph 워크플로우 구성
workflow = StateGraph(AwakePlusState)
workflow.add_node("scrape", scrape_awakeplus)
workflow.add_node("filter", filter_new_disclosures)
workflow.add_node("dart_fetch", fetch_dart_details)

workflow.add_edge(START, "scrape")
workflow.add_edge("scrape", "filter")
workflow.add_edge("filter", "dart_fetch")
workflow.add_edge("dart_fetch", END)

awakeplus_app = workflow.compile()
