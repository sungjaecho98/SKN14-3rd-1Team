from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

load_dotenv()

PINECONE_INDEX_NAME = 'supplement-rag'
PINECONE_INDEX_REGION = 'us-east-1'
PINECONE_INDEX_CLOUD = 'aws'
PINECONE_INDEX_METRIC = 'cosine'
PINECONE_INDEX_DIMENSION = 1536

OPENAI_LLM_MODEL = 'gpt-4o-mini'
OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'

# Pinecone 초기화
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# LangChain용 벡터스토어 연결
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)

# 사용자 질문 처리 함수
llm = ChatOpenAI(temperature=0, model_name=OPENAI_LLM_MODEL)

def ask_nutrition_question(question: str) -> str:
    retriever = vector_store.as_retriever(
        search_type="mmr", # 다양성과 유사도를 동시에 고려하는 전략
        search_kwargs={"k": 3, "lambda_mult": 0.7}
    )

    docs_with_scores = vector_store.similarity_search_with_score(question, k=3)
    threshold = 0.8
    relevant_docs = [doc for doc, score in docs_with_scores if score >= threshold]

    # TODO 계속 llm 기반 검색만 함...
    if relevant_docs:
        print("vectordb 기반 검색")
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever).run(question)
    else:
        print("llm 기반 검색")
        return llm.invoke(question).content