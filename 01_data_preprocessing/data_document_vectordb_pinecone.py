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

    for entry in api_json['body']['items']:
        data = entry['item']

        def get_and_strip(data_dict, key):  # strip()을 바로 쓰면 문자열이 아닐 경우 오류 발생하므로
            value = data_dict.get(key)
            return value.strip() if isinstance(value, str) else ''

        text = f"""
제품명: {get_and_strip(data, 'PRDUCT')}
제조사: {get_and_strip(data, 'ENTRPS')}
기능성: {get_and_strip(data, 'MAIN_FNCTN')}
섭취 시 주의사항: {get_and_strip(data, 'INTAKE_HINT1')}
보관조건: {get_and_strip(data, 'PRSRV_PD')}
유통기한: {get_and_strip(data, 'DISTB_PD')}
"""

        metadata = {
            "등록일자": data.get("STTEMNT_NO"),
            "제조사": data.get("ENTRPS"),
            "기준규격": get_and_strip(data, "BASE_STANDARD")
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents

# ------------------------
# 2. 전체 페이지 반복 요청하여 모든 문서 수집
# ------------------------
def fetch_all_documents(api_url, api_key, num_of_rows=100) -> list[Document]:
    all_documents = []
    params = {
        "ServiceKey": api_key, # API 키
        "pageNo": "1", # 시작 페이지
        "numOfRows": str(num_of_rows), # 한 페이지에 수집할 문서 수
        "type": "json", # 응답 형식
    } # API 요청 파라미터 설정

    # 일단 첫 페이지 요청
    response = requests.get(api_url, params=params, timeout=10) # API 요청
    response.raise_for_status() # 응답 상태 코드 확인
    first_page = response.json() # 첫 페이지 응답 JSON 파싱

    total_count = int(first_page['body']['totalCount']) # 전체 문서 수 카운트, 1페이지 응답을 통해 전체 문서 수 파악
    total_pages = (total_count // num_of_rows) + (1 if total_count % num_of_rows else 0) # 전체 페이지 수 계산

    # 이미 받은 first_page를 활용해서 문서로 바꾸고 리스트에 추가
    all_documents.extend(json_to_documents(first_page)) 

    for page in range(2, total_pages + 1):
        params['pageNo'] = str(page) # 현재 페이지 번호를 API 요청 파라미터에 설정
        response = requests.get(api_url, params=params, timeout=10) # 해당 페이지에 대한 API 요청전송
        response.raise_for_status() # 오류가 있으면 예외 발생시킴 (예: 404, 500)
        page_json = response.json() # 페이지 응답 JSON 파싱
        docs = json_to_documents(page_json) # 해당 페이지의 JSON을 Document 객체로 변환
        all_documents.extend(docs) # 변환된 문서들을 all_documents 리스트에 추가
        print(f"📄 {page}/{total_pages} 페이지 수집 완료") 

    print(f"\n✅ 총 {len(all_documents)}개의 Document 객체 생성 완료")
    return all_documents

# ------------------------
# 3. 벡터스토어 생성 및 문서 업로드
# ------------------------
def build_vector_store(documents, index_name):
    embeddings = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)

    # Pinecone 초기화 및 인덱스 준비
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

    # 배치 업로드
    batch_size = 100
    for i in range(0, len(split_documents), batch_size):
        batch = split_documents[i:i + batch_size]
        # vector_store.add_documents(batch)  ####### 주석 해제 시 실제 업로드 수행
        print(f"Added batch {i//batch_size + 1}/{(len(split_documents)//batch_size) + 1}")

    print("\n✅ All documents added to Pinecone vector store.")
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
if __name__ == "__main__":
    url = 'http://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01'
    api_key = os.environ.get('API_KEY')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    # Step 1: 문서 수집
    documents = fetch_all_documents(url, api_key)

    # Step 2: 벡터스토어 생성 및 문서 업로드
    index_name = 'health-supplement-rag'
    vector_store = build_vector_store(documents, index_name)

    # Step 3: 질의 응답 시스템 구축
    qa_chain = build_qa_chain(vector_store)

    # Step 4: 사용자 질의 처리
    query = "피로개선에 도움이 되는 영양제는?"
    response = qa_chain(query)

    print("\n🧠 GPT 응답:")
    print(response['result'])

    print("\n📎 참고 문서:")
    for doc in response['source_documents']:
        # print(doc.page_content)
        print(f"제품명: {doc.metadata['제품명']} | 제조사: {doc.metadata['제조사']} | 등록일자: {doc.metadata['등록일자']}")
        print("-" * 60)