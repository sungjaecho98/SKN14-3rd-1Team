# 개선된 PromptTemplate (RAG + Fallback + 사용자 친화)
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


# 1. RAG용 프롬프트
rag_prompt = PromptTemplate.from_template("""
당신은 건강기능식품 및 영양제 추천 전문가입니다.

다음은 건강기능식품 관련 문서입니다:

{context}

위 문서를 바탕으로 아래 질문에 최대한 정확하고 신뢰성 있게 답변해주세요.
질문: "{question}"

- 반드시 문서의 내용을 기반으로만 답변하세요.
- 효과나 효능이 불확실한 내용은 언급하지 마세요.
- 사용자에게 도움이 될 수 있는 건강기능식품을 구체적으로 추천하세요.
- 말투는 친절하고 상냥하되, 정보는 정확하게 제공하세요.

※ 답변 마지막에 다음 문장을 붙이세요:
건강기능식품은 의약품이 아닙니다. 전문가와 상담하세요.
""")

# 2. fallback 프롬프트 (검색 결과 없을 때) -> 모델 비싼 거 쓰니까 답변 잘 해준다해서 이거 필요없어짐
fallback_prompt = PromptTemplate.from_template("""
"{question}"에 대한 관련 정보를 찾을 수 없습니다.

정확한 정보를 제공하기 어려우므로, 섣부른 추천은 하지 않겠습니다.

건강기능식품은 의약품이 아닙니다. 전문가와 상담하세요.
""")

# 3. LLM 세팅
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-4.1-mini",
    temperature=0.3,
    max_tokens=512
)

# 4. QA 체인 (stuff 방식)
from langchain.chains.question_answering import load_qa_chain
qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=rag_prompt)

# 5. fallback 체인
fallback_chain = LLMChain(llm=llm, prompt=fallback_prompt)

# 6. 최종 실행 체인 (문서 검색 → 있으면 QA, 없으면 fallback)
def custom_rag_executor(question: str):
    # 문서 검색
    docs = retriever.get_relevant_documents(question)

    if docs:
        # 검색 결과 있음 → QA 수행
        return qa_chain.run(input_documents=docs, question=question)
    else:
        # 검색 결과 없음 → fallback 수행
        return fallback_chain.run({"question": question})

# 7. 질문 실행
query = "똥이 안 나와"
response = custom_rag_executor(query)
print("\n챗봇 응답:")
print(response)
