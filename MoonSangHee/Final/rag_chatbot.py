from config import load_config
import os, json
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

        self.openai_api_key = os.getenv('OPENAI_API_KEY')
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
        # self.prompt = self.prompt()


    def run(self, question, temperature=0.3, max_token=1024): 

        
        llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=temperature, model_name=self.openai_model_name, max_tokens=max_token)
        retrieved_docs = self.retriever.get_relevant_documents(question)
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        prompt_template = self.prompt(question=question, context=context)

        # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)
        # output = qa_chain.invoke({"query": question})

        response = llm.invoke(prompt_template.format(question=question, context=context))

        # return {'answer': response.content, 'source_documents': retrieved_docs}
        return response.content                        

    def prompt(self, question, context):
        system_prompt = PromptTemplate.from_template(f"""
        
         [System Instruction]
        - 당신은 여러문서를 분석하여 사용자의 질문에 친절히 답변하는 건강기능식품 전문가이다.
        - 만약, 증상을 입력하면 증상에 대한 공감을 먼저 한 뒤, 제품을 추천해줘야 한다.
        - 추천한 제품은 자세하게 설명해줘야 한다.
        - 제품을 추천할 시, 3개의 제품을 추천해야 한다. 만약, 3개의 제품이 없다면, 있는 만큼만 추천해준다
        - 답변은 반드시 [Example - Output Indicator]에 따라야 한다.
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

# cfg = load_config()    
# rag = RAG_Chatbot(cfg)
# question = '비타민 중에 임산부도 먹을 수 있는 제품 추천해줘'
# retriever = rag.retriever
# res = rag.run(question, retriever, cfg['OPENAI_MODEL_NAME'])
# print(res['answer'])
# # pprint(retriever.invoke('피로개선에 도움이 되는 영양제는?'))