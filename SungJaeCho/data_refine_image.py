# SungJaeCho/data_refine.py

import os
import time
import requests
from pprint import pprint
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain.chains import RetrievalQA

# ------------------------
# 1. API 응답 -> LangChain Document 변환 함수
# ------------------------
def json_to_documents(api_json: dict) -> list[Document]:
    documents = []

    # 네이버 API 키 환경변수에서 가져오기
    naver_client_id = os.environ.get("NAVER_CLIENT_ID")
    naver_client_secret = os.environ.get("NAVER_CLIENT_SECRET")

    for entry in api_json['body']['items']:
        data = entry['item']

        def get_and_strip(data_dict, key):
            value = data_dict.get(key)
            return value.strip() if isinstance(value, str) else ''

        product_name = get_and_strip(data, 'PRDUCT')
        image_url = ""
        if naver_client_id and naver_client_secret and product_name:
            image_url = fetch_image_url(product_name, naver_client_id, naver_client_secret)

        text = f"""
제품명: {product_name}
제조사: {get_and_strip(data, 'ENTRPS')}
기능성: {get_and_strip(data, 'MAIN_FNCTN')}
섭취 시 주의사항: {get_and_strip(data, 'INTAKE_HINT1')}
보관조건: {get_and_strip(data, 'PRSRV_PD')}
유통기한: {get_and_strip(data, 'DISTB_PD')}
"""

        metadata = {
            "제품명": product_name,
            "등록일자": data.get("STTEMNT_NO"),
            "제조사": data.get("ENTRPS"),
            "기준규격": get_and_strip(data, "BASE_STANDARD"),
            "이미지URL": image_url
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents

# ------------------------
# 2. 전체 페이지 반복 요청하여 모든 문서 수집
# ------------------------
# def fetch_all_documents(api_url, api_key, num_of_rows=100) -> list[Document]:
#     all_documents = []
#     params = {
#         "ServiceKey": api_key, # API 키
#         "pageNo": "1", # 시작 페이지
#         "numOfRows": str(num_of_rows), # 한 페이지에 수집할 문서 수
#         "type": "json", # 응답 형식
#     } # API 요청 파라미터 설정

#     # 일단 첫 페이지 요청
#     response = requests.get(api_url, params=params, timeout=10) # API 요청
#     response.raise_for_status() # 응답 상태 코드 확인
#     first_page = response.json() # 첫 페이지 응답 JSON 파싱

#     total_count = int(first_page['body']['totalCount']) # 전체 문서 수 카운트, 1페이지 응답을 통해 전체 문서 수 파악
#     total_pages = (total_count // num_of_rows) + (1 if total_count % num_of_rows else 0) # 전체 페이지 수 계산

#     # 이미 받은 first_page를 활용해서 문서로 바꾸고 리스트에 추가
#     all_documents.extend(json_to_documents(first_page)) # .extend(docs)는 all_documents에 문서들을 낱개로 차곡차곡 넣는 역할

#     for page in range(2, total_pages + 1):
#         params['pageNo'] = str(page) # 현재 페이지 번호를 API 요청 파라미터에 설정
#         response = requests.get(api_url, params=params, timeout=10) # 해당 페이지에 대한 API 요청전송
#         response.raise_for_status() # 오류가 있으면 예외 발생시킴 (예: 404, 500)
#         page_json = response.json() # 페이지 응답 JSON 파싱
#         docs = json_to_documents(page_json) # 해당 페이지의 JSON을 Document 객체로 변환
#         all_documents.extend(docs) # 변환된 문서들을 all_documents 리스트에 추가
#         print(f"📄 {page}/{total_pages} 페이지 수집 완료") 

#     print(f"\n✅ 총 {len(all_documents)}개의 Document 객체 생성 완료")
#     return all_documents

def fetch_all_documents(api_url, api_key, num_of_rows=100, test_pages=1) -> list[Document]:
    all_documents = []
    params = {
        "ServiceKey": api_key,
        "pageNo": "1",
        "numOfRows": str(num_of_rows),
        "type": "json",
    }

    response = requests.get(api_url, params=params, timeout=10)
    response.raise_for_status()
    first_page = response.json()

    total_count = int(first_page['body']['totalCount'])
    total_pages = (total_count // num_of_rows) + (1 if total_count % num_of_rows else 0)

    # 테스트: test_pages 만큼만 수집
    all_documents.extend(json_to_documents(first_page))

    for page in range(2, min(test_pages, total_pages) + 1):
        params['pageNo'] = str(page)
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        page_json = response.json()
        docs = json_to_documents(page_json)
        all_documents.extend(docs)
        print(f"📄 {page}/{total_pages} 페이지 수집 완료 (테스트)")

    print(f"\n✅ (테스트) 총 {len(all_documents)}개의 Document 객체 생성 완료")
    return all_documents

# ------------------------
# 3. 벡터스토어 생성 및 문서 업로드
# ------------------------
# def build_vector_store(documents, index_name):
#     embeddings = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])

#     # 문서 분할
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     split_documents = text_splitter.split_documents(documents)

#     # Pinecone 초기화 및 인덱스 준비
#     pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=embeddings.get_dimension(),
#             metric='cosine'
#         )
#         while not pc.describe_index(index_name).status['ready']:
#             time.sleep(1)

#     vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

#     # 배치 업로드
#     batch_size = 100
#     for i in range(0, len(split_documents), batch_size):
#         batch = split_documents[i:i + batch_size]
#         # vector_store.add_documents(batch)  # 주석 해제 시 실제 업로드 수행
#         print(f"Added batch {i//batch_size + 1}/{(len(split_documents)//batch_size) + 1}")

#     print("\n✅ All documents added to Pinecone vector store.")
#     return vector_store
def build_vector_store(documents, index_name):
    embeddings = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)

    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embeddings.get_dimension(),
            metric='cosine'
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    batch_size = 100
    for i in range(0, len(split_documents), batch_size):
        batch = split_documents[i:i + batch_size]
        # 실제 업로드는 주석 처리 (테스트)
        vector_store.add_documents(batch)
        print(f"(테스트) Added batch {i//batch_size + 1}/{(len(split_documents)//batch_size) + 1}")

    print("\n✅ (테스트) All documents processed for Pinecone vector store.")
    return vector_store


# ------------------------
# 4. RAG 질의 시스템 구성
# ------------------------
def build_qa_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 3}
    )

    llm = ChatOpenAI(
        model_name="gpt-4.1-mini",
        temperature=0.3,
        max_tokens=512
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain

# ------------------------
# 5. 메인 실행부 예시
# ------------------------
# if __name__ == "__main__":
#     url = 'http://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01'
#     api_key = os.environ.get('OPENAI_API_KEY')

#     # Step 1: 문서 수집
#     documents = fetch_all_documents(url, api_key)

#     # Step 2: 벡터스토어 생성 및 문서 업로드
#     index_name = 'health-supplement-rag'
#     vector_store = build_vector_store(documents, index_name)

#     # Step 3: 질의 응답 시스템 구축
#     qa_chain = build_qa_chain(vector_store)

#     # Step 4: 사용자 질의 처리
#     query = "피로개선에 도움이 되는 영양제는?"
#     response = qa_chain(query)

#     print("\n🧠 GPT 응답:")
#     print(response['result'])

#     print("\n📎 참고 문서:")
#     for doc in response['source_documents'][:3]:
#         print(f"제품명: {doc.metadata['제품명']} | 제조사: {doc.metadata['제조사']} | 등록일자: {doc.metadata['등록일자']}")
#         print(f"이미지URL: {doc.metadata.get('이미지URL', '')}")
#         print("-" * 60)

if __name__ == "__main__":
    url = 'http://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01'
    api_key = os.environ.get('OPENAI_API_KEY')

    # Step 1: 문서 수집 (테스트: 2페이지만 수집)
    documents = fetch_all_documents(url, api_key, num_of_rows=10, test_pages=2)

    # Step 2: 벡터스토어 생성 및 문서 업로드 (실제 업로드는 주석 처리)
    index_name = 'health-supplement-rag-test'
    vector_store = build_vector_store(documents, index_name)

    # Step 3: 질의 응답 시스템 구축
    qa_chain = build_qa_chain(vector_store)

    # Step 4: 사용자 질의 처리
    query = "피로개선에 도움이 되는 영양제는?"
    response = qa_chain(query)

    print("\n🧠 GPT 응답:")
    print(response['result'])

    print("\n📎 참고 문서:")
    for doc in response['source_documents'][:3]:
        print(f"제품명: {doc.metadata['제품명']} | 제조사: {doc.metadata['제조사']} | 등록일자: {doc.metadata['등록일자']}")
        print(f"이미지URL: {doc.metadata.get('이미지URL', '')}")
        print("-" * 60)


# 네이버 이미지 검색 API 호출 함수
def fetch_image_url(product_name, client_id, client_secret):
    import urllib.parse

    base_url = "https://openapi.naver.com/v1/search/image"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    params = {
        "query": product_name,
        "display": 1,
        "sort": "sim"
    }
    response = requests.get(base_url, headers=headers, params=params, timeout=5)
    if response.status_code == 200:
        items = response.json().get("items")
        if items:
            return items[0].get("link")
    return ""

