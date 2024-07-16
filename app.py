import json
import os
import sys
import boto3
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.llms import Bedrock
from langchain_huggingface import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings =  HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import S3DirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
from langchain_community.vectorstores import FAISS

## Langchain Integration
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")


## Data ingestion
def data_ingestion():
    loader = S3DirectoryLoader(bucket="ganeshbedrockbucket1", prefix="ramayana/" )
    documents=loader.load()
    print(documents)

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

## LLM

def get_mistral_llm():
    llm=Bedrock(model_id="mistral.mistral-7b-instruct-v0:2",client=bedrock,
                model_kwargs={'max_tokens':512})
    
    return llm


prompt_template = """

    Human: Use the following pieces of context to provide a concise answer to the question provided.
    Limit your answer to 250 words. If you lack context, or don't know about something, respond with a message saying you don't know!

    <context>
    {context}
    </context>

    Question : {question}

    Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def add_custom_css():
    st.markdown("""
        <style>
            .stApp {
                background-color: #f0f2f6;
            }
            .header {
                font-size: 36px;
                font-weight: bold;
                color: #4B4B4B;
                text-align: center;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            .subheader {
                font-size: 24px;
                color: #4B4B4B;
                text-align: center;
                margin-top: 10px;
                margin-bottom: 10px;
            }
            .sidebar .sidebar-content {
                background-color: #fff;
                border-radius: 10px;
                padding: 20px;
            }
            .sidebar .sidebar-content .title {
                font-size: 20px;
                font-weight: bold;
                color: #4B4B4B;
            }
            .stButton button {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton button:hover {
                background-color: #45a049;
            }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    add_custom_css()

    st.markdown("<div class='header'>Chat with PDF using AWS Bedrock!</div>", unsafe_allow_html=True)

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    with st.sidebar:
        st.markdown("<div class='title'>Update Or Create Vector Store:</div>", unsafe_allow_html=True)

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated successfully!")

    if st.button("Get Answer"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            llm = get_mistral_llm()
            
            #faiss_index = get_vector_store(docs)
            answer = get_response_llm(llm, faiss_index, user_question)
            st.write(answer)
            st.success("Done")

if __name__ == "__main__":
    main()