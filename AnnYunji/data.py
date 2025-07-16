import requests
import os
import math
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from pinecone import Pinecone

load_dotenv()

API_KEY = os.getenv('API_KEY')
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
PINECONE_INDEX_NAME = "health-supplement-rag"
url = 'http://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01?'

def fetch_all_documents(rows_per_page=100, max_pages=None) -> list[Document]:
    documents = []

    params = {
        "pageNo": "1",
        "numOfRows": str(rows_per_page),
        "ServiceKey": API_KEY,
        "type": "json"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    body = response.json()["body"]
    total_count = int(body.get("totalCount", 0))
    total_pages = math.ceil(total_count / rows_per_page)

    if max_pages:
        total_pages = min(total_pages, max_pages)

    for page in range(1, total_pages + 1):
        params["pageNo"] = str(page)
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        body_ = resp.json()["body"]
        items_section = body_.get("items", [])

        if isinstance(items_section, list):
            items_list = [entry["item"] for entry in items_section]
        else:
            items_list = [items_section["item"]]

        for data in items_list:
            text = f"""
제품명: {(data.get('PRDUCT') or '').strip()}
제조사: {(data.get('ENTRPS') or '').strip()}
기능성: {(data.get('MAIN_FNCTN') or '').strip()}
섭취 시 주의사항: {(data.get('INTAKE_HINT1') or '').strip()}
보관조건: {(data.get('PRSRV_PD') or '').strip()}
유통기한: {(data.get('DISTB_PD') or '').strip()}
"""

            metadata = {
                "등록일자": data.get("STTEMNT_NO"),
                "제조사": data.get("ENTRPS"),
                "기준규격": (data.get("BASE_STANDARD") or '').strip()
            }

            documents.append(Document(page_content=text, metadata=metadata))

        print(f"{page}페이지: {len(items_list)}개 처리 완료")
        time.sleep(0.3)

    return documents


from pinecone import Pinecone, ServerlessSpec

pc = Pinecone()
print(pc.list_indexes().names())

PINECONE_INDEX_NAME = 'health-supplement-rag'
PINECONE_INDEX_REGION = 'us-east-1'
PINECONE_INDEX_CLOUD = 'aws'
PINECONE_INDEX_METRIC = 'cosine'
PINECONE_INDEX_DIMENSION = 1536

OPENAI_LLM_MODEL = 'gpt-4.1-nano'
OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_INDEX_DIMENSION,
        metric=PINECONE_INDEX_METRIC,
        spec=ServerlessSpec(
            region=PINECONE_INDEX_REGION,
            cloud=PINECONE_INDEX_CLOUD,
        )
    )
    print(f'{PINECONE_INDEX_NAME} index 생성완료!')
else:
    print(f'{PINECONE_INDEX_NAME} index가 이미 존재합니다.')

def embed_and_upload_documents(documents, chunk_size=1000, chunk_overlap=200, batch_size=100):
    print("문서 분할 시작")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    print(f"총 {len(split_docs)}개의 청크로 분할 완료")

    # 임베딩 모델 불러오기
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

    # 벡터 스토어 연결
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
    )

    print("Pinecone 업로드 시작")
    for i in tqdm(range(0, len(split_docs), batch_size)):
        batch = split_docs[i:i + batch_size]
        try:
            vector_store.add_documents(batch)
        except Exception as e:
            print(f"{i}번째 배치에서 에러 발생: {e}")

    print("모든 문서 업로드 완료.")

if __name__ == "__main__":
    all_documents = fetch_all_documents()
    print(f"총 {len(all_documents)}개의 문서를 수집했습니다.")

    embed_and_upload_documents(all_documents)
