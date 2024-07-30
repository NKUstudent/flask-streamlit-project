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
from streamlit_lottie import st_lottie
from streamlit_chat import message
import streamlit as st
from Web_virtual_character import SourceService, ChatModel

# 读取 Lottie JSON 文件
def load_lottiefile(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

# 允许用户上传多个文件并保存
def upload_save():
    # 创建一个文件夹用来保存上传的文件
    dir = 'D:\\uploads'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # 选择上传文件
    uploaded_files = st.file_uploader('请选择待上传的文件：', accept_multiple_files = True, type = ['pdf', 'txt', 'docx', 'csv'])
    # 保存上传文件
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            file_contents = uploaded_file.getvalue()
            file_path = os.path.join(dir, uploaded_file.name)
            # 将文件存到本地的文件系统中
            with open(file_path, 'wb') as f:
                f.write(file_contents)
        return dir

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

# 页面选择
def main_page():
    # Render the Lottie animation

    st_lottie(
        lottie_animation_01,
        speed=1,  # 设置动画速度为正常播放速度
        reverse=False,
        loop=True,
        quality="high",
        height=None,  # 让动画适应容器大小
        width=None,  # 让动画适应容器大小
        key="lottie_animation",
    )
    col1, col2 , col3, col4, col5 = st.columns(5)
    with col3:
        # Render the button
        if st.button(':blue[Try now !]'):
            st.session_state.page = 'page'
            st.experimental_rerun()

def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
    st.session_state.generated.append("The messages from Bot\nWith new line")

def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]

def show_page():
    col1,col2,col3,col4,col5 = st.columns(5)
    with col3:
        st.title("NULL Chats")
    #st.file_uploader("Upload a story", type=["txt"], key="file_uploader")

    with st.container(border=True):

        if 'past' not in st.session_state:
            st.session_state['past'] = []
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        
        #st.session_state.setdefault(
        #    'past',
        #    ['请告诉我《红楼梦》里贾宝玉的故事情节。',
        #     '能为我生成一个林黛玉的画像吗？', ]
        #)
        #st.session_state.setdefault(
        #    'generated',
        #    [{'type': 'normal',
        #      'data': '贾宝玉是《红楼梦》的主角，他是贾府的少爷，自幼与许多表姐妹一起长大，尤其与林黛玉关系密切。他性格叛逆，对仕途经济毫无兴趣，最后在家庭败落、亲人离世的打击下，离家出走，成为了一个僧人。'},
        #     {'type': 'normal',
        #      'data': '当然可以。林黛玉是一个美丽但体弱多病的女子，她有着一双水汪汪的大眼睛和一头乌黑的长发，常穿着淡雅的衣裙，气质清冷。'
        #              f'<img width="100%" height="200" src="{img_path}"/>'}]
        #)

        chat_placeholder = st.empty()

        with chat_placeholder.container():
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
                message(
                    st.session_state['generated'][i]['data'],
                    key=f"{i}",
                    allow_html=True,
                    is_table=True if st.session_state['generated'][i]['type'] == 'table' else False
                )

        col1, col2 = st.columns([5,1])
        with col1:
            with st.container():
                text = st.text_area("User Input:", on_change=on_input_change, key="user_input",height=50)
        with col2:
            with st.container():
                st.write("\n")
                st.write('\n')
                st.write('\n')
                st.button("Clear message", on_click=on_btn_click)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col3:
        if st.button(':blue[Back]'):
            st.session_state.page = 'main'
            st.experimental_rerun()
    return text
    
def chat(input):
    data = {'query': input}
    response = send_input_to_flask(data)
    st.session_state['past'].append(input)
    st.session_state['generated'].append(response)

if __name__ == '__main__':
    # 加载 Lottie 动画
    lottie_animation_01 = load_lottiefile("main_01.json")
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    file_address = upload_save()
    file_path = {'address': file_address}
    send_file_address_to_flask(file_path)
    while(True):
        text = page()
        chat(input)