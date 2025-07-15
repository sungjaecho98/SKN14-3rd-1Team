from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import RegexParser
from rag import new_docs, vector_store, retriever
from config import load_config


def build_rag_chain(system_prompt, llm_model = "gpt-4o-mini", temperature = 0.3):

    llm = ChatOpenAI(openai_api_key=cfg["OPENAI_API_KEY"], model=llm_model, temperature=temperature)

    parser_regex: str = r"(?s)(.*)"
    parser = RegexParser(regex=parser_regex, output_keys=["answer"])

    chain = system_prompt | llm | parser

    return chain


def run_query(query, retriever, chain):

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    output = chain.invoke({
        "context": context,
        "question": query})

    return output["answer"]

def prompt():
    system_prompt = PromptTemplate.from_template("""
    [System Instruction]
    - 당신은 여러문서를 분석하여 사용자의 질문에 친절히 답변하는 건강기능식품 전문가이다.
    - 만약, 증상을 입력하면 증상에 대한 공감을 먼저 한 뒤, 제품을 추천해줘야 한다.
    - 추천한 제품은 자세하게 설명해줘야 한다.
    - 제품을 추천할 시, 3개의 제품을 추천해야 한다. 만약, 3개의 제품이 없다면, 있는 만크만 추천해준다
    - 답변은 반드시 [Example - Output Indicator]에 따라야 한다.
    - 주어진 문서내에서만 정보를 추출해 답변해야 한다.
    - 사용자의 질문에 대한 내용을 주어진 문서상에서 찾을 수 없는 경우 찾을수 없다고 답변해야 한다.
    - 절대 말을 지어내어서는 안된다!!!
    - ~것으로 끝나지 않도록 문장으로 답변해야 한다.

    [Context]
    {context}

    [Input Data]
    {question}

    [Example - Output Indicator]
    Q: 11종 혼합유산균의 유통기한은?
    A: 제조일로부터 24개월입니다.

    Q: 11종 혼합유산균의 섭취시 주의사항은?
    A: 1. 질환이 있거나 의약품 복용 시 전문가와 상담하십시오.
    2. 알레르기 체질 등은 개인에 따라 과민반응을 나타낼 수 있습니다.
    3. 어린이가 함부로 섭취하지 않도록 일일섭취량 방법을 지도해 주십시오.
    4. 이상사례 발생 시 섭취를 중단하고 전문가와 상담하십시오.

    """)
    return system_prompt

cfg = load_config()
file_path = './output_cleaned.jsonl'

retriever = retriever(file_path)
system_prompt = prompt()
chain = build_rag_chain(system_prompt=system_prompt)
query = '쓰리플렉스 콜레스테롤은 임산부가 섭취해도 돼?'
res = run_query(query, retriever, chain)

print(res)

