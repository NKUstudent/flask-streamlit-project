from drawpicture import draw
from PIL import Image
import requests
from flask import request
import pandas as pd
import json
import io
from io import StringIO
import tempfile
from pathlib import Path
import base64
import chardet
import os
import shutil

import streamlit as st

# 发送数据到 flask 后端中
def send_data_to_flask(data):
    url = 'http://127.0.0.1:5000/submit_data'
    response = requests.post(url, json = data)
    return response.json()

# 创建用户输入对话信息页面
def chat_windows():
    option = st.radio(
        label = '请输入你想进行的操作：',
        options = ('对话', '绘图')
    )

    st.title('Input your Query')
    text = st.text_area(
        label = '请输入文本',
        height = 5,
        max_chars = 200,
        help = '输入的最大长度限制为200'
    )

    if option == '绘图':
        img_url = draw(text)
        response = requests.get(img_url)
        image = Image.open(io.BytesIO(response.content))
        st.image(
            image,
            caption = 'picture',
            width = 400
        )
    else:
        data = {'query': text}
        response = send_data_to_flask(data)
        st.write('Response from Flask')
        st.write(response)

# 允许用户上传多个文件并保存
def upload_save():
    # 创建一个文件夹用来保存上传的文件
    dir = 'D:\\uploads'
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)
    # 选择上传文件
    uploaded_files = st.file_uploader('请选择待上传的文件：', accept_multiple_files = True, type = ['pdf', 'txt', 'docx', 'csv'])
    # 保存上传文件
    file_address = []
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            file_contents = uploaded_file.getvalue()
            file_path = os.path.join(dir, uploaded_file.name)
            # 将文件存到本地的文件系统中
            with open(file_path, 'wb') as f:
                f.write(file_contents)
            st.write(f"文件地址: {file_path}")
            file_address.append(file_path)
        return file_address

def show_file():
    uploaded_file = st.file_uploader('choose a file:', type = 'pdf')
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete = False) as tmp_file:
            fp = Path(tmp_file.name)
            fp.write_bytes(uploaded_file.getvalue())
            with open(tmp_file.name, 'rb') as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" ' \
                                f'width="800" height="1000" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == '__main__':
    chat_windows()
    upload_save()