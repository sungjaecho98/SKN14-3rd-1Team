#%%
#!pip install langchain langchain-community langchain-pinecone langchain-openai pypdf
#%% md
# api 받아오기
#%%
import requests
from pprint import pprint
import time
from dotenv import load_dotenv
import os

end_point = 'https://apis.data.go.kr/1471000/HtfsInfoService03'

url = 'http://apis.data.go.kr/1471000/HtfsInfoService03/getHtfsItem01?'

load_dotenv()
decoding = os.getenv['DECOIDNG']

# all_items = []  # 전체 데이터를 모을 리스트
# page = 1        # 시작 페이지
#
# while True:
#     # 요청 변수 세팅
#     params = {
#         "pageNo": str(page),
#         "numOfRows": "100",  # API 최대 허용값
#         "ServiceKey": "decoding",
#         "type": "json",
#     }
#
#     # API 요청
#     response = requests.get(url, params=params, timeout=10)
#     response.raise_for_status()
#
#     # 결과 확인
#     data = response.json()
#     body = data.get('body')
#     if not body:
#         print(f"{page}페이지 - body 없음 종료")
#         break
#
#     items = body.get('items')
#     if not items:
#         print(f"{page}페이지 - items 없음 종료")
#         break
#
#     all_items.extend(items)
#     print(f"{page}페이지 수집 완료 - 누적 {len(all_items)}개")
#
#     page += 1
#
#
# #수집된 데이터를 파일로 저장
# with open('raw_data.json', 'w', encoding='utf-8') as f:
#     json.dump(all_items, f, ensure_ascii=False, indent=4)
#
# print(f"총 {len(all_items)}개 데이터 저장 완료")
#


num_of_rows = 100
page = 1

# 3) 요청 변수(params) 세팅
params = {
    "pageNo":    "1",          # 페이지 번호
    "numOfRows": str(num_of_rows),         # 한 페이지 결과 수
    "ServiceKey": os.getenv('DECODING_KEY'), # Swagger에 표시된 정확한 이름(ServiceKey)
    "type":      "json",       # 응답 포맷(xml/json) – default: xml
    # (필요시 API별 추가 파라미터를 여기에 더합니다)
}

# 4) 실제 요청 보내기
response = requests.get(url, params=params, timeout=10)
response.raise_for_status()

# 5) 결과 확인
pprint(response.json())
#%%
def extract_product_info(api_response):
    items = api_response['body']['items']

    simplified_products = []

    for entry in items:
        item = entry['item']
        simplified_products.append({
            '제품명': item.get('PRDUCT', '').strip(),
            '기능성': item.get('MAIN_FNCTN', '').strip(),
            '제조사': item.get('ENTRPS', '').strip(),
            '섭취방법': item.get('SRV_USE', '').strip(),
            '보관방법': item.get('PRSRV_PD', '').strip(),
            '주의사항': item.get('INTAKE_HINT1', '').strip(),
        })

    return simplified_products

product_list = extract_product_info(response.json())

# 출력문 (예쁘게 정리)
for i, p in enumerate(product_list[:3], 1):  # TOP3만 예시
    print(f"[{i}] {p['제품명']}")
    print(f"기능성: {p['기능성']}")
    print(f"제조사: {p['제조사']}")
    print(f"섭취방법: {p['섭취방법']}")
    print(f"보관방법: {p['보관방법']}")
    print(f"주의사항: {p['주의사항']}")
    print("-" * 50)
#%%
print(len(product_list))
#%% md
# document생성
#%%
from langchain.schema import Document

def convert_to_documents(product_list):
    docs = []

    for product in product_list:
        content = (
            f"제품명: {product['제품명']}\n"
            f"기능성: {product['기능성']}\n"
            f"섭취방법: {product['섭취방법']}\n"
            f"보관방법: {product['보관방법']}\n"
            f"주의사항: {product['주의사항']}"
        )

        metadata = {
            "제품명": product["제품명"], # 중복으로 넣음
            "제조사": product["제조사"],
        }

        docs.append(Document(page_content=content, metadata=metadata))

    return docs

documents = convert_to_documents(product_list)
#%% md
# 임베딩
#%%
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# 임베딩 모델
embeddings = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])

# 벡터스토어 객체(client) 생성
vector_store = PineconeVectorStore.from_documents(
    documents,
    embedding=embeddings,
    index_name='healthcare'
)
#%% md
# Pinecone 저장
#%%
from langchain.vectorstore import pinecone


#%% md
# 쿼리 날려
#%%
