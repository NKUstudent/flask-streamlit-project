# 导包
import os
import json
import requests
import threading
import streamlit as st
from langchain_community.chat_models import ChatZhipuAI

# 环境变量设置
import modelchoice
modelchoice.setenv()

os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0'
os.environ['SERPAPI_API_KEY'] = 'a3a020895a0debde83c3a13b048ce4c3fb97151dcd1ccd1769ed0a8a8abe1858'

# 构建语言模型
chat_model = ChatZhipuAI(model = 'glm-4')

# 1. 读取本地文件
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
local_dir = 'C:\\Users\\墨池洗砚\\Desktop\\学习\\暑期实训\\WorkSpace\\langchain-demo\\documents'
documents = []

# 遍历documents文件夹下文件
for filename in os.listdir(local_dir):
    file_path = os.path.join(local_dir, filename)
    if filename.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif filename.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif filename.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())
    elif filename.endswith('.csv'):
        loader = CSVLoader(file_path)
        documents.extend(loader.load())

# 2. 从网页上爬取数据
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    # 导入相关网站，可导入多个
    web_paths = ['https://sciencebasedmedicine.org/what-is-traditional-chinese-medicine/'] 
)
# 整合本地文件数据与网页数据
documents.extend(loader.load())

# 3. 将数据存储进向量数据库 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 切分文档数据
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 1)
chunked_documents = text_splitter.split_documents(documents = documents)
# 使用嵌入模型
EMBEDDING_DEVICE = 'cpu'
embeddings = HuggingFaceEmbeddings(model_name = '..\小组项目\m3e-base',
                                   model_kwargs = {'device': EMBEDDING_DEVICE})
# 建立索引：将词向量存储到向量数据库
vector = FAISS.from_documents(documents = chunked_documents, embedding = embeddings)
retriever = vector.as_retriever()

# 4. 提示词工程
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import (
                                    MessagesPlaceholder,
                                    ChatPromptTemplate,
)
history_prompt = ChatPromptTemplate.from_messages([
    ('system', '根据以上对话历史，生成一个检索查询，以便查找与对话相关的信息'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human', '{input}'),
])
#生成含有历史信息的检索链
retriever_history_chain = create_history_aware_retriever(chat_model, retriever, history_prompt)
# 继续对话 记住检索到的文档等信息
totol_prompt = ChatPromptTemplate.from_messages([
    ('system', "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human', '{input}')
])

# 5. 生成 Retrieval 模型
from langchain.chains.combine_documents import create_stuff_documents_chain
document_chain = create_stuff_documents_chain(chat_model, totol_prompt)
from langchain.chains.retrieval import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever_history_chain, document_chain)

# 6. 构建 flask 框架 | 生成 WebUI QA
from flask import Flask, jsonify, request

app = Flask(__name__)
@app.route('/submit_data', methods = ['POST'])
def submit_data():
    data = request.json
    print(f'Received data: {data}')
    return jsonify({'status': 'success', 'message': data}), 200

if __name__ == '__main__':
    app.run(debug = True, port = 5000)