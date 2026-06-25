import os
import datetime
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# 분석할 이미지 디렉토리
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "01_raw", "Test")
# ChromaDB 저장 경로
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def ingest_images():
    print("우가! 임베딩 모델(CLIP) 불러온다!")
    model = SentenceTransformer('clip-ViT-B-32')
    
    print(f"우가! ChromaDB 초기화한다! ({DB_DIR})")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name="image_rag_cosine",
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"우가! 테이블 준비 완료. 폴더({IMAGE_DIR}) 탐색 시작!")
    
    if not os.path.exists(IMAGE_DIR):
        print(f"경고: {IMAGE_DIR} 경로가 존재하지 않는다 우가!")
        return

    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                filepath = os.path.join(root, file)
                
                try:
                    # 이미지 열고 임베딩 생성
                    img = Image.open(filepath)
                    
                    # 1. 크기(size) 추출
                    width, height = img.size
                    
                    # 2. 날짜(date) 추출 (파일 수정 시간 기준)
                    mtime = os.path.getmtime(filepath)
                    date_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    
                    # CLIP 모델은 이미지와 텍스트를 동일한 차원의 벡터로 반환함
                    vector = model.encode(img).tolist()
                    
                    # DB에 저장 (upsert: 있으면 덮어쓰기, 없으면 추가)
                    collection.upsert(
                        ids=[filepath],
                        embeddings=[vector],
                        documents=[f"이 이미지는 {file} 입니다 우가!"],
                        metadatas=[{
                            "filename": file,
                            "width": width,
                            "height": height,
                            "date": date_str
                        }]
                    )
                    print(f"저장 성공: {file} (크기: {width}x{height}, 날짜: {date_str})")
                except Exception as e:
                    print(f"에러 발생 ({file}): {e}")

    print("우가! 데이터 준비 싹 다 끝났다!")

if __name__ == "__main__":
    ingest_images()
