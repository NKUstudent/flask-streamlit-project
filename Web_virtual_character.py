# 导包
import os
import json
import requests
import threading
import chardet
import shutil
import streamlit as st
from config import LangchainCFG
from flask import Flask, jsonify, request
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import (
                                    MessagesPlaceholder,
                                    ChatPromptTemplate,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import DirectoryLoader

# 环境变量设置
import modelchoice
modelchoice.setenv()

os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'
os.environ['SERPAPI_API_KEY'] = 'a3a020895a0debde83c3a13b048ce4c3fb97151dcd1ccd1769ed0a8a8abe1858'

class Web_interface(object):
    def __init__(self, chat_model):
        self.chat_model = chat_model

    def chat(self):
        messages = st.container(height = 600)
        user_input = st.chat_input('请输入...')
        if user_input:
            messages.chat_message('user').write(user_input)
            messages.chat_message('assistant').write(self.chat_model.get_llm_answer(user_input))
        with st.sidebar:
            st.toast(user_input)


class SourceService(object):
    def __init__(self, config):
        self.config = config
        self.vector_store = None
        self.embeddings =  HuggingFaceEmbeddings(model_name = self.config.embedding_model_name,
                                                 model_kwargs = {'device': 'cpu'})
        self.vector_store_path = self.config.vector_store_path
    
    # 切分源文件
    def split_source_file(self):
        if os.path.exists(self.config.vector_store_path):
            shutil.rmtree(self.config.vector_store_path)
        if os.path.exists(self.config.doc_path):
            shutil.rmtree(self.config.doc_path)
        os.makedirs(self.config.vector_store_path)
        os.makedirs(self.config.doc_path)
    
        endcoding = ''
        with open(self.config.source_file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            endcoding = result['encoding']
            if endcoding == 'GB2312':
                endcoding = 'GB18030'

        open_txt = open(self.config.source_file_path, 'r', encoding = endcoding)
        txt_line = open_txt.readlines()
        line_list = []
        for line in txt_line:
            line_list.append(line)
        count = len(line_list)
    
        # 切分txt文件
        txt_split = [line_list[i:i+20] for i in range(0, count, 20)]
        # 将切分的数据写入多个txt中
        for i, j in zip(range(0, int(count/20+1)), range(0, int(count/20+1))):
            with open('D:\\uploads\\《红楼梦》%d.txt' % j, 'w+', encoding = endcoding) as temp:
                for line in txt_split[i]:
                    temp.write(line)
    
    # 初始化向量数据库
    def init_source_vector(self):
        documents = []
        text_loader_kwargs = {'autodetect_encoding': True}
        loader = DirectoryLoader(self.config.doc_path, glob = "**/*.txt", loader_cls = TextLoader, loader_kwargs = text_loader_kwargs)
        documents.extend(loader.load())
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)

    # 上传本地的向量数据库
    def load_vector_store(self, path):
        if path is not None:
            self.vector_store = FAISS.load_local(path, self.embeddings)
        else:
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
        return self.vector_store
    
    # 从网页上爬取数据
    def search_web(self, web_path):
        loader = WebBaseLoader(web_path)
        web_data = loader.load()
        self.vector_store.add_documents(web_data)
        self.vector_store.save_local(self.vector_store_path)

class ChatModel(object):
    def __init__(self, config):
        self.config = config
        self.llm = self.config.chat_llm
        self.source_service = SourceService(self.config)
        self.source_service.split_source_file()
        self.source_service.init_source_vector()
        self.history = []
        
    # 创建模型框架
    def create_llm(self):
        # 提示词工程
        history_prompt = ChatPromptTemplate.from_messages([
            ('system', '根据以上对话历史，生成一个检索查询，以便查找与对话相关的信息'),
            MessagesPlaceholder(variable_name = 'history'),
            ('human', '{input}'),
        ])
        # 生成含有历史信息的检索链
        retriever_history_chain = create_history_aware_retriever(self.llm, self.source_service.vector_store.as_retriever(), history_prompt)
        # 继续对话 记住检索到的文档等信息
        totol_prompt = ChatPromptTemplate.from_messages([
            ('system', "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name = 'history'),
            ('human', '{input}')
        ])
        # 生成 Retrieval 模型
        document_chain = create_stuff_documents_chain(self.llm, totol_prompt)
        retrieval_chain = create_retrieval_chain(retriever_history_chain, document_chain)
        return retrieval_chain

    def get_llm_answer(self, query):
        chat_ai = self.create_llm()
        result = chat_ai.invoke({
            'history': self.history,
            'input': query,
        })
        answer = result['answer']
        self.history.append(HumanMessage(content = query))
        self.history.append(AIMessage(content = answer))
        return answer


if __name__ == '__main__':
    config = LangchainCFG()
    chat_model = ChatModel(config)
    page = Web_interface(chat_model)
    page.chat()
