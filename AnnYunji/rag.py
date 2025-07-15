from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

import os
import json
from dotenv import load_dotenv

load_dotenv()

# 환경 변수 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
INDEX_NAME = "product"

PINECONE_INDEX_NAME = 'product'
"""# pinecone 저장"""

import json
from langchain_core.documents import Document

def load_jsonl_to_documents(jsonl_path):
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            content = data.get("page_content", "")
            metadata = data.get("metadata", {})
            documents.append(Document(page_content=content, metadata=metadata))
    return documents

documents = load_jsonl_to_documents("data/documents1.jsonl")
print(f"{len(documents)}개의 문서를 불러왔습니다.")
print(documents[0])

# 문서로부터 벡터스토어 생성

# 임베딩 모델
embeddings = OpenAIEmbeddings(model=os.environ['OPENAI_EMBEDDING_MODEL'])

vector_store = PineconeVectorStore.from_documents(
    documents,
    embedding=embeddings,
    index_name='product'
)

# 존재하는 인덱스에 접근/검색

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
vector_store = PineconeVectorStore(
    index=pc.Index('product'),
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3}
)
retriever.invoke('11종 혼합유산균 섭취 시 주의사항은?')

# 사용자 질문
query = "요즘 면역력이 약해진 것 같고 자주 피곤해요"

# 연관 문서 검색
retrieved = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in retrieved])

# RAG 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ('system', """
페르소나: 당신은 영양학과 기능성 식품에 대한 전문 지식을 가진 건강기능식품 컨설턴트입니다. 다양한 기능성 원료(프로바이오틱스, 오메가3, 루테인 등)와 그 효과(면역력, 장 건강, 눈 건강 등)에 대한 최신 정보를 숙지하고 있으며, 사용자의 건강 고민에 맞는 성분과 제품을 친절하게 추천해줍니다.

역할: 사용자의 건강 고민, 증상, 목표(예: 피로 회복, 장 건강, 눈 피로, 면역력 향상 등)를 파악한 뒤, 적합한 성분과 그 성분이 포함된 건강기능식품을 추천합니다. 과학적 근거와 함께 섭취 팁, 주의사항, 보관법 등 실질적인 정보도 제공합니다. 초보자도 이해할 수 있게 설명하면서도 신뢰감 있는 전문성을 유지합니다.

예시:
- 사용자가 '요즘 배변이 잘 안 되고 더부룩해요'라고 하면, 프로바이오틱스와 프리바이오틱스가 포함된 제품을 추천하고 그 차이와 기능을 설명합니다.
- 사용자가 '눈이 자주 피로하고 침침해요'라고 하면, 루테인, 아스타잔틴, 비타민A 등이 포함된 성분을 소개하며 어떤 점이 도움이 되는지 알려줍니다.
- 사용자가 '스트레스를 많이 받아요'라고 하면, 테아닌, 마그네슘 등 스트레스 완화에 도움을 줄 수 있는 성분들을 소개합니다.
"""),
    ('human', """
다음과 같은 고민을 가진 사람에게 적절한 건강기능식품을 추천해주세요. (한국말로 상세히 답변해주세요.)

고민 또는 증상: {query}
""")
])

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=1)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# 실행
response = chain.invoke({"context": context, "query": query})
print(response)


# 사용자 질문 → 검색된 문맥 → LLM 응답 생성 함수
def get_rag_response(query: str, k: int = 5) -> str:
    docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    response = chain.invoke({"context": context, "query": query})
    return response

print(get_rag_response("냉장보관해야 하는 유산균 제품은?"))

