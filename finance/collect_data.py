import os
import pandas as pd
import FinanceDataReader as fdr
from opendartreader import OpenDartReader
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import datetime
from langchain_ollama import ChatOllama
from tqdm import tqdm

# 환경변수 로드
load_dotenv()

# API 클라이언트 초기화
tavily_api_key = os.environ.get("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

dart_api_key = os.environ.get("DART_API_KEY")
dart = OpenDartReader(dart_api_key) if dart_api_key else None

# 로컬 LLM 초기화
llm = ChatOllama(model="qwen2.5:1.5b")

def get_kospi_metadata(limit=5):
    # 폴더 구조 설정
    base_dir = "data/01_raw/KOSPI"
    run_time = datetime.now()
    folder_name = run_time.strftime("%y%m%d_%H%M")
    run_dir = os.path.join(base_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    
    meta_path = os.path.join(base_dir, "meta.xlsx")
    
    # 기존 메타데이터 로드
    if os.path.exists(meta_path):
        meta_df = pd.read_excel(meta_path)
    else:
        meta_df = pd.DataFrame(columns=[
            "종목코드", "종목명", "현재가", "현재가 업데이트 날짜", 
            "공시 업데이트 날짜", "공시 정보 파일 경로", 
            "주요뉴스 .md 경로", "뉴스 업데이트 날짜"
        ])
        
    tqdm.write("KOSPI 목록 가져오는 중...")
    kospi_list = fdr.StockListing('KOSPI')
    
    if limit:
        kospi_list = kospi_list.head(limit)
    
    # tqdm 진행률 표시기
    pbar = tqdm(kospi_list.iterrows(), total=len(kospi_list), desc="KOSPI 수집")
    
    for idx, row in pbar:
        code = row['Code']
        name = row['Name']
        pbar.set_description(f"[{name}]")
        
        # 1. 현재 가격
        pbar.set_postfix(작업="현재가 조회")
        current_price = None
        price_update_date = None
        try:
            df_price = fdr.DataReader(code)
            if not df_price.empty:
                current_price = df_price['Close'].iloc[-1]
                price_update_date = df_price.index[-1].strftime("%Y-%m-%d")
        except Exception as e:
            tqdm.write(f"[{name}] 현재가 에러: {e}")
            
        # 2. 최근 공시 수집
        pbar.set_postfix(작업="공시 조회")
        dart_date = None
        dart_file_path = None
        if dart:
            try:
                dart_res = dart.list(code)
                if dart_res is not None and not dart_res.empty:
                    dart_date = dart_res['rcept_dt'].iloc[0]
                    dart_file_path = os.path.join(run_dir, f"disclosures_{code}.xlsx").replace("\\", "/")
                    dart_res.to_excel(dart_file_path, index=False)
            except Exception as e:
                tqdm.write(f"[{name}] 공시 수집 에러: {e}")
                
        # 3. 뉴스 긁어오기 및 LLM 분석
        pbar.set_postfix(작업="뉴스 검색(Tavily)")
        news_file_path = None
        news_update_date = run_time.strftime("%Y-%m-%d %H:%M")
        
        if tavily_client:
            try:
                res = tavily_client.search(query=f"{name} 주식", max_results=10)
                news_results = res.get('results', [])
                
                if news_results:
                    md_lines = [f"# {name} 주요 뉴스 분석 ({news_update_date})", ""]
                    news_context = ""
                    md_lines.append("## 뉴스 출처")
                    for i, r in enumerate(news_results, 1):
                        title = r.get('title', '')
                        url = r.get('url', '')
                        content = r.get('content', '')
                        md_lines.append(f"{i}. [{title}]({url})")
                        news_context += f"제목: {title}\n내용: {content}\n\n"
                        
                    md_lines.append("")
                    md_lines.append("## LLM 분석 결과 (qwen2.5:1.5b)")
                    
                    prompt = (
                        f"다음은 '{name}' 관련 최신 뉴스 10개의 요약 내용입니다.\n"
                        f"{news_context}\n"
                        f"위 내용을 바탕으로 주요 내용을 요약하고, 해당 소식들이 주가에 전반적으로 호재인지, 악재인지, 중립인지 명확하게 분류해주세요."
                    )
                    
                    pbar.set_postfix(작업="LLM 분석 중")
                    llm_res = llm.invoke(prompt)
                    llm_text = llm_res.content if hasattr(llm_res, 'content') else str(llm_res)
                    md_lines.append(llm_text)
                    
                    pbar.set_postfix(작업="마크다운 저장")
                    news_file_path = os.path.join(run_dir, f"news_{code}.md").replace("\\", "/")
                    with open(news_file_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(md_lines))
                else:
                    tqdm.write(f"[{name}] 검색된 뉴스가 없습니다.")
                    
            except Exception as e:
                tqdm.write(f"[{name}] 뉴스/LLM 에러: {e}")
                
        # 4. 메타데이터 업데이트
        pbar.set_postfix(작업="메타데이터 업데이트")
        new_row = {
            "종목코드": code,
            "종목명": name,
            "현재가": current_price,
            "현재가 업데이트 날짜": price_update_date,
            "공시 업데이트 날짜": dart_date,
            "공시 정보 파일 경로": dart_file_path,
            "주요뉴스 .md 경로": news_file_path,
            "뉴스 업데이트 날짜": news_update_date
        }
        
        idx = meta_df.index[meta_df['종목코드'] == code].tolist()
        if idx:
            for k, v in new_row.items():
                meta_df.loc[idx[0], k] = v
        else:
            meta_df = pd.concat([meta_df, pd.DataFrame([new_row])], ignore_index=True)
            
    # 최종 메타데이터 저장
    meta_df.to_excel(meta_path, index=False)
    tqdm.write(f"\n수집 완료! 메타데이터가 {meta_path} 에 저장되었습니다.")

def main():
    get_kospi_metadata(limit=5)

if __name__ == "__main__":
    main()
