from dotenv import load_dotenv
import os

load_dotenv()

def load_config():
    
    config = {}

    config['OPENAI_EMBEDDING_MODEL'] = 'text-embedding-3-small'
    config['OPENAI_MODEL_NAME'] = 'gpt-4.1-mini'

    # config['PINECONE_ENV'] = 'us-east-1'
    # config['VECTOR_STORE_INDEX_NAME'] = 'health-supplement-rag'

    config['FAISS_FILE_PATH'] = 'faiss_index'

    return config
