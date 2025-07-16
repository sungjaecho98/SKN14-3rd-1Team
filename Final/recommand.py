from langchain_tavily import TavilySearch
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

def get_recommendation_from_web(query: str, cfg):
    # 1. Pinecone 연결
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embeddings = OpenAIEmbeddings(model=cfg['OPENAI_EMBEDDING_MODEL'])
    vectorstore = PineconeVectorStore(
        index_name=cfg['VECTOR_STORE_INDEX_NAME'],
        embedding=embeddings,
    )

    # 2. 유사 제품 검색
    retriever = vectorstore.as_retriever()
    similar_products = retriever.get_relevant_documents(query, k=3)

    if not similar_products:
        return '[{"error": "Pinecone에서 유사 제품을 찾지 못했습니다."}]'

    # 3. 제품명만 추출
    product_names = []
    for doc in similar_products:
        page = doc.page_content
        name = ""

        for line in page.splitlines():
            if line.startswith("제품명:"):
                name = line.replace("제품명:", "").strip()

        # 혹시 누락된 경우 metadata에서 보완
        if not name:
            name = doc.metadata.get("제품명", "")

        if name:
            product_names.append(name)

    if not product_names:
        return '[{"error": "제품명을 추출할 수 없습니다."}]'

    # 4. Tavily 검색용 쿼리 구성 (각 제품 하나씩 질의)
    web_query = "\n".join(
        [f"{p} 제품에 대한 정보와 단독 이미지 알려줘" for p in product_names]
    )

    # 5. Tavily Tool 및 LLM 준비
    llm = ChatOpenAI(model=cfg['OPENAI_MODEL_NAME'], temperature=0)
    tavily_tool = TavilySearch(max_results=2, include_images=True)
    tools = [tavily_tool]

    # 6. ReAct 프롬프트 설정
    REACT_PROMPT = """
당신은 건강기능식품 전문 AI입니다. 아래 도구를 사용하여 사용자의 질문에 가장 적절한 제품 정보를 제공합니다.

{tools}

다음 형식을 따르세요:

Question: 사용자의 질문
Thought: 어떤 조치를 취할지 생각
Action: 사용할 도구 (다음 중 하나 [{tool_names}])
Action Input: 도구에 입력할 내용
Observation: 도구의 출력 결과
... (이 과정을 반복하여 필요한 정보를 수집)
Thought: 이제 최종 답변을 알겠어요
Final Answer: 실제 제품 이미지가 존재하고, 제품명과 완벽히 일치하는 정확한 2개의 제품 정보를 JSON 리스트로 반환

**규칙 및 조건**
- 반드시 JSON 배열 형태로 응답하세요 (설명 금지)
- 모든 항목은 **한국어**로 작성
- 포함 필드:
  - `"name"`: 제품명 (정확히 표기)
  - `"brand"`: 브랜드명 (정확히 표기)
  - `"price"`: "숫자원" 형식 (예: `"32000원"`)
  - `"category"`: 카테고리 또는 효능
  - `"rating"`: 0.0 ~ 5.0 사이 실수
  - `"reviews"`: 리뷰 수 (정수)
  - `"image_url"`: 제품 단독 이미지의 URL
  - `'ingredients'`: 성분 리스트
  - `'benefits'`: 효과 리스트
  - `'dosage'`: 복용법 설명
  - `'warnings'`: 주의사항 리스트

**출력 예시**

Final Answer:
[
  {{
    "name": "고려홍삼정 로얄",
    "brand": "정관장",
    "price": "42000원",
    "category": "면역력 증진",
    "rating": 4.7,
    "reviews": 253,
    "image_url": "https://example.com/product1.jpg",
    "ingredients": ["홍삼", "꿀"],
    "benefits": ["면역력 증진", "피로 회복"],
    "dosage": "하루 1회, 1포씩 섭취",
    "warnings": ["임산부는 섭취 전 의사와 상담", "과다 섭취 주의"]
  }},
  {{
    "name": "프로바이오틱스 플러스",
    "brand": "종근당건강",
    "price": "27000원",
    "category": "장 건강 개선",
    "rating": 4.5,
    "reviews": 191,
    "image_url": "https://example.com/product2.jpg"
  }}
]

Question: {input}
{agent_scratchpad}
"""

    prompt = PromptTemplate(
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        template=REACT_PROMPT
    )

    # 7. Agent 생성 및 실행
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    result = agent_executor.invoke({"input": web_query})
    return result.get("output", "추천 결과 없음")