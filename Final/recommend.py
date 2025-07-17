from langchain_tavily import TavilySearch
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json
import requests

load_dotenv()


def search_image_google(product_name: str, brand: str) -> str:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CX = os.getenv("GOOGLE_CSE_ID")

    query = f"{brand} {product_name}"
    params = {
        "q": query,
        "cx": CX,
        "key": GOOGLE_API_KEY,
        "searchType": "image",
        "num": 3,
        "imgSize": "medium",
    }

    response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
    data = response.json()

    # print(f"\n--- Google API 응답 (쿼리: {query}) ---")
    # print(json.dumps(data, indent=2, ensure_ascii=False))
    # print("-------------------------------------------\n")

    brand_lower = brand.lower()
    product_name_parts = [k.lower() for k in product_name.split()]
    meaningful_name_keywords = [
        k for k in product_name_parts
        if k != brand_lower and len(k) > 1
    ]

    if not meaningful_name_keywords and product_name:
        meaningful_name_keywords = [product_name.lower()]
    elif not meaningful_name_keywords:
        meaningful_name_keywords = [brand_lower]

    for item in data.get("items", []):
        image_url = item.get("link", "")
        if not image_url:
            continue

        context_link = item.get("image", {}).get("contextLink", "")
        item_title = item.get("title", "")

        filter_target_string = (context_link + " " + item_title).lower()

        # 브랜드명이 contextLink 또는 title에 포함되는지 확인
        brand_match = brand_lower in filter_target_string

        # 핵심 키워드 중 몇 개나 일치하는지 계산
        keyword_matches_count = sum(1 for keyword in meaningful_name_keywords if keyword in filter_target_string)

        print(f"URL: {image_url}, 타이틀: {item_title}, 브랜드 매치: {brand_match}, 키워드 매치 수: {keyword_matches_count}")

        # 브랜드 일치 여부와 관계없이 핵심 키워드 2개 이상 일치 시 허용
        if (brand_match and (keyword_matches_count >= 1 or not meaningful_name_keywords)) or (keyword_matches_count >= 2):
            return image_url

    return ""


def get_recommendation_from_web(query: str, cfg):
    llm = ChatOpenAI(model=cfg['OPENAI_MODEL_NAME'], temperature=0)
    tavily_tool = TavilySearch(max_results=2, include_images=False)
    tools = [tavily_tool]

    # 커스텀 ReAct 프롬프트 설정
    REACT_PROMPT = """
    당신은 건강기능식품 전문가 AI입니다. 아래 도구들을 이용해 질문에 가장 적절한 답변을 제공하세요:

    {tools}

    다음 형식을 사용하세요:

    Question: 사용자의 질문
    Thought: 어떤 조치를 취할지 생각. 먼저, Tavily 검색을 통해 필요한 정보를 얻고, **그 결과가 실제 온라인 쇼핑몰에서 판매되는 제품인지 교차 검증해야 합니다.**
    Action: 사용할 도구 (다음 중 하나 [{tool_names}])
    Action Input: 도구에 입력할 내용
    Observation: 도구의 출력 결과
    ... (이 Thought/Action/Observation 과정은 반복 가능)
    Thought: 이제 최종 답변을 알겠어요
    Final Answer: 질문에 대한 최종 답변 (JSON 형식으로 정확히 2개의 제품을 반환)

    - 최종 응답은 반드시 **설명 없이 JSON 리스트 형식**으로, **모든 항목은 한국어로 작성**해주세요.  
    - 가격은 숫자형 + "원" 단위로 표기하고, `"price": "32000원"`처럼 문자열로 반환해주세요.

    - **'name' 필드에 대한 지침: 제품의 '정확한 상호명(full product name)'을 포함해야 합니다. "임산부 종합비타민"과 같이 포괄적인 이름이 아닌, 실제 온라인 쇼핑몰(예: 쿠팡, 네이버 스마트스토어 등)에서 검색했을 때 **가장 잘 찾아지는 고유한 제품명**을 찾아 반영하세요.**
    - ** 최우선 중요 지침: **
        - Final Answer에 포함될 모든 제품 정보는 반드시 온라인 쇼핑몰(예: "쿠팡", "네이버 스마트스토어", "G마켓", "옥션", "SSG닷컴", "올리브영" 등)에서 **현재 활발히 판매 중이며 그 존재가 명확히 확인된 실제 제품**만을 기반으로 해야 합니다.  
        - Tavily 검색 결과를 통해 얻은 정보가 실제 판매 중인 제품과 일치하는지 **반드시 교차 검증**하고, 제품의 **정확한 이름과 브랜드**를 확인하세요.  
        - 환각성(Hallucination) 제품 정보는 절대 제공해서는 안 됩니다!  
        - 만약 Tavily 검색 결과가 불확실하거나 실제 제품을 찾기 어렵다면, 해당 제품을 추천 목록에서 제외하고 다음으로 적합한 실제 제품을 찾으세요.**

    - 주의: 이 단계에서는 "image_url" 필드를 비워 두세요. 이미지는 후처리 단계에서 Google API를 통해 자동으로 채워집니다.
        - 각 제품의 `"image_url"` 필드는 빈 문자열("") 또는 null로 남겨두세요.
        - 이 필드는 나중에 Google 이미지 검색 API를 통해 보완됩니다.
        - 이 단계에서는 절대 이미지를 포함하지 마세요.

    - 예시 형식:
    Final Answer:
    [
      {{
        "name": "홍삼활력보", 
        "brand": "정관장",
        "price": "32000원",
        "ingredients": ["성분1", "성분2"],
        "benefits": ["효과1", "효과2"],
        "dosage": "복용법 설명",
        "warnings": ["주의사항1", "주의사항2"],
        "category": "카테고리명",
        "rating": 4.5,
        "reviews": 123,
        "image_url": ""
        }},
        ...
    ]

    Question: {input}
    {agent_scratchpad}
    """

    prompt = PromptTemplate(
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        template=REACT_PROMPT
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    result = agent_executor.invoke({"input": query})

    # JSON 파싱
    try:
        product_list = json.loads(result["output"])
    except Exception as e:
        print("JSON 파싱 오류:", e)
        return result["output"]

    # 이미지 URL 후처리
    for product in product_list:
        name = product.get("name", "")
        brand = product.get("brand", "")
        new_image = search_image_google(name, brand)
        if new_image:
            product["image_url"] = new_image
        else:
            product["image_url"] = None

        print(f"제품: {name}, 브랜드: {brand}, 이미지 URL: {product['image_url']}")

    valid_products = [p for p in product_list if p["image_url"]]
    return valid_products
