from config import load_config
import os, json
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain.chains import RetrievalQA
from pprint import pprint
from ocr_llm import OCR_LLM

class RAG_Chatbot():

    def __init__(self, cfg):

        super().__init__()

        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')

        self.cfg = cfg
        self.ocr = OCR_LLM(self.cfg)

        self.index_name = self.cfg['VECTOR_STORE_INDEX_NAME']
        self.openai_embedding_model = self.cfg["OPENAI_EMBEDDING_MODEL"]
        self.pinecone_env = self.cfg['PINECONE_ENV']
        self.openai_model_name = self.cfg['OPENAI_MODEL_NAME']

        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=self.openai_embedding_model )
        self.pc = Pinecone(api_key=self.pinecone_api_key, environment=self.pinecone_env)
        self.vector_store = PineconeVectorStore(index=self.pc.Index(self.index_name), embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3}) 
        # self.prompt = self.prompt()


    def run(self, question="", use_ocr='False', img_file=None, temperature=0.3, max_token=1024): 

        
        llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=temperature, model_name=self.openai_model_name, max_tokens=max_token)
  
        if use_ocr:

            if img_file is None:
                raise ValueError("OCR 모드가 활성화되어 있으나 이미지 파일이 제공되지 않았습니다. 이미지를 업로드한 후 다시 시도해 주세요.")
        
            try:
                question = str(self.ocr.ocr_to_llm(img_file)).strip()
            except Exception as e:
                raise RuntimeError(f"OCR 처리 중 예상치 못한 오류가 발생했습니다: {e}")

            retrieved_docs = self.retriever.get_relevant_documents(question)
            context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
            prompt_template = self.prompt_ocr(question=question, context=context)
        
        else:
            retrieved_docs = self.retriever.get_relevant_documents(question)
            context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
            prompt_template = self.prompt(question=question, context=context)

            
        response = llm.invoke(prompt_template.format(question=question, context=context))

        return response.content                        

    def prompt(self, question, context):
        system_prompt = PromptTemplate.from_template(f"""
        
         [System Instruction]
        - 당신은 여러문서를 분석하여 사용자의 질문에 친절히 답변하는 건강기능식품 전문가이다.
        - 입력된 키워드들을 조합해 해당 문서에서 가장 유사한 제품을 찾아야한다.
        - 찾은 제품은 자세하게 설명해준다.
        - 주어진 문서내에서만 정보를 추출해 답변해야 한다.
        - 사용자의 질문에 대한 내용을 주어진 문서상에서 찾을 수 없는 경우 찾을수 없다고 답변해야 한다.
        - 절대 말을 지어내어서는 안된다!!!
        - ~것으로 끝나지 않도록 문장으로 답변해야 한다.

        [Context]
        {context}

        [Input Data]
        {question}

        """)
        return system_prompt
    
    def prompt_ocr(self, question, context):

        prompt = PromptTemplate.from_template(f"""
            [System Instruction]
            당신은 여러 문서를 분석하여 사용자의 질문에 친절히 답변하는 건강기능식품 전문가입니다.

            입력된 키워드가 문서에서 일부라도 포함된 유사한 건강기능식품 3종을 찾습니다.
            키즈, 유아는 같은 의미입니다.

            응답 시 유의사항:
            - 반드시 주어진 문서 내 정보만을 기반으로 답변하세요.
            - 정보를 찾을 수 없는 경우, "해당 문서에서 찾을 수 없습니다."라고 답변하세요.
            - 정보를 찾은 경우 아래 항목을 포함하여 문장을 평서형으로 작성하세요:
            
            1. 제품명 및 브랜드
            2. 기대 효과 및 기능성
            3. 섭취 방법                                   
            4. 주요 성분 및 함량
            5. 섭취 시 주의사항
            - 절대 말을 지어내거나 문서를 벗어난 내용을 포함하지 마세요.

            # OCR 키워드 입력
            {question}

            # 문서 내용
            {context}

            # 답변:
            """)
        
        return prompt
cfg = load_config()
rag = RAG_Chatbot(cfg)
res = rag.run(use_ocr=True, img_file='./image7.jpeg')
print(res)