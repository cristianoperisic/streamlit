import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone 설정 (콘솔에서 만든 인덱스 이름)
PINECONE_INDEX_NAME = "rm-project-index"

# 유효성 검사
if not OPENAI_API_KEY:
    raise ValueError("🚨 OPENAI_API_KEY가 .env 파일에 없습니다.")
if not PINECONE_API_KEY:
    raise ValueError("🚨 PINECONE_API_KEY가 .env 파일에 없습니다.")
