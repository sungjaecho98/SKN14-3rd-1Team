from dotenv import load_dotenv
import os

load_dotenv()

def load_config():
    
    config = {}

    config['FILE_PATH'] = './output_cleaned.jsonl'

    config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY_2')
    config['OPENAI_EMBEDDING_MODEL'] = os.getenv('OPENAI_EMBEDDING_MODEL')

    config['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    config['PINECONE_ENV'] = os.getenv('PINECONE_ENV')
    config['VECTOR_STORE_INDEX_NAME'] = 'project3'

    config['LANGSMITH_TRACING'] = os.getenv('LANGSMITH_TRACING')
    config['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT')
    config['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
    config['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')

    return config
