from config import load_config
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Pinecone
import os, json
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain.chains import RetrievalQA
from pprint import pprint

class RAG_Chatbot():

    def __init__(self, cfg):

        super().__init__()

        self.openai_api_key = os.getenv('OPENAI_API_KEY_2')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')

        self.cfg = cfg

        self.index_name = self.cfg['VECTOR_STORE_INDEX_NAME']
        self.openai_embedding_model = self.cfg["OPENAI_EMBEDDING_MODEL"]
        self.pinecone_env = self.cfg['PINECONE_ENV']
        self.openai_model_name = self.cfg['OPENAI_MODEL_NAME']

        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=self.openai_embedding_model )
        self.pc = Pinecone(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        self.vector_store = PineconeVectorStore(index=self.pc.Index(self.index_name), embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3}) 
        self.prompt = self.prompt()


    def run(self, question, retriever, model_name, temperature=0.3, max_token=512): 

        
        llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=temperature, model_name=model_name, max_tokens=max_token)

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)
        output = qa_chain.invoke({"query": question})

        return output
                                 

    def prompt(self):
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
rag = RAG_Chatbot(cfg)
question = '아, 똥이 안나와'
retriever = rag.retriever
res = rag.run(question, retriever, cfg['OPENAI_MODEL_NAME'])
print(res)
# pprint(retriever.invoke('피로개선에 도움이 되는 영양제는?'))