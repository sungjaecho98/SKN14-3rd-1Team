from config import load_config
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import os, json
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

cfg = load_config()
file_path = './output_cleaned.jsonl'

def new_docs(file_path):
    new_docs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            rec = json.loads(raw)
            parts = []

            for k, v in rec.items():
                val = "" if v is None else str(v)
                parts.append(f"{k}: {val}")
            combined_text = "\n".join(parts)

            new_docs.append(Document(page_content=combined_text, metadata=rec))

    return new_docs


def vector_store(docs, embeddings, index_name, mode='read'):

    pc = Pinecone(api_key=cfg['PINECONE_API_KEY'], environment=cfg['PINECONE_ENV'])
    indexes_meta = pc.list_indexes()
    index_names = indexes_meta.names()

    if index_name not in index_names:

        pc.create_index(name=index_name,dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region=cfg['PINECONE_ENV']))
        vector_store = PineconeVectorStore.from_documents(docs, embedding=embeddings, index_name=index_name)

    else:

        if mode == 'read':
            vector_store = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)

        elif mode == 'write':
            vector_store = PineconeVectorStore.from_documents(docs, embedding=embeddings, index_name=index_name)
        
    return vector_store


def retriever(docs_file_path, search_type='mmr', k=3, lambda_mult=0.7, vector_store_mode='read'):

    docs = new_docs(docs_file_path)

    embeddings = OpenAIEmbeddings(openai_api_key=cfg["OPENAI_API_KEY"], model=cfg["OPENAI_EMBEDDING_MODEL"] )
    vector_stores = vector_store(docs, embeddings, index_name=cfg['VECTOR_STORE_INDEX_NAME'], mode=vector_store_mode)

    retriever = vector_stores.as_retriever(search_type=search_type, search_kwargs={'k': k, 'lambda_mult': lambda_mult})

    return retriever, vector_stores

