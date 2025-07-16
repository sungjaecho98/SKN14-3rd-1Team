from langchain_tavily import TavilySearch
from langchain.agents import create_react_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def get_recommendation_from_web(query: str, cfg):
    llm = ChatOpenAI(model=cfg['OPENAI_MODEL_NAME'], temperature=0)
    tavily_tool = TavilySearch(max_results=3)
    tools = [tavily_tool]

    # 🔧 커스텀 ReAct 프롬프트 설정
    REACT_PROMPT = """
당신은 건강기능식품 전문가 AI입니다. 아래 도구들을 이용해 질문에 가장 적절한 답변을 제공하세요:

{tools}

다음 형식을 사용하세요:

Question: 사용자의 질문
Thought: 어떤 조치를 취할지 생각
Action: 사용할 도구 (다음 중 하나 [{tool_names}])
Action Input: 도구에 입력할 내용
Observation: 도구의 출력 결과
... (이 Thought/Action/Observation 과정은 반복 가능)
Thought: 이제 최종 답변을 알겠어요
Final Answer: 질문에 대한 최종 답변 (JSON 형식으로 정확히 3개의 제품을 반환)

- 최종 응답은 반드시 **설명 없이 JSON 리스트 형식**으로, **모든 항목은 한국어로 작성**해주세요.  
- 가격은 숫자형 + "원" 단위로 표기하고, `"price": 32000원`처럼 문자열로 반환해주세요.

- 예시 형식:

Final Answer:
[
  {{
    "name": "제품명",
    "brand": "브랜드명",
    "price": "32000원",
    "ingredients": ["성분1", "성분2"],
    "benefits": ["효과1", "효과2"],
    "dosage": "복용법 설명",
    "warnings": ["주의사항1", "주의사항2"],
    "category": "카테고리명",
    "rating": 4.5,
    "reviews": 123
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
    return result["output"]
