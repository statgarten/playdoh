import streamlit as st
import datetime
from PIL import Image
from sklearn.datasets import load_iris
import pandas as pd
from streamlit_cropper import st_cropper
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model

def number(wine_df1):
    col1, col2, col3 = st.columns(3)
    with col1:
        col1_1, col1_2 = st.columns(2)
        col1_1.header('최대 : ')
        col1_2.header(max(wine_df1))

    with col2:
        col2_1, col2_2 = st.columns(2)
        col2_1.header('최소 : ')
        col2_2.header(min(wine_df1))

    with col3:
        col3_1, col3_2 = st.columns(2)
        col3_1.header('평균 : ')
        col3_2.header(round(sum(wine_df1)/len(wine_df1), 2))
        
st.set_page_config(layout="wide")
option = st.sidebar.selectbox('',('이미지 분류', '텍스트 분류', 'Speech2Text', '시계열 데이터 예측'))

if option == '이미지 분류':
    
    st.header('이미지 분류')
    col1, col2, col3, col4, col5 = st.columns([10, 3, 5, 3, 6])
    with col1:
        col1_1, col1_2 = st.columns([2, 5])
        col1_1.write('Class1')
        rnna = col1_2.button(':pencil2:')
            
        img_file = st.file_uploader(label='Upload a file', type=['png', 'jpg','webp'])
        if img_file is not None:
            col1_1_1,col1_1_2,col1_1_3 = st.columns(3)
            realtime_update = col1_1_1.checkbox(label="Update in Real Time", value=True)
            box_color = col1_1_2.color_picker(label="Box Color", value='#0000FF')
            aspect_choice = col1_1_3.radio(label="Aspect Ratio", options=["1:1", "16:9", "Free"])
            
            aspect_dict = {
                "1:1": (1, 1),
                "16:9": (16, 9),
                "Free": None
            }
            aspect_ratio = aspect_dict[aspect_choice]

            if img_file:
                img = Image.open(img_file)
                if not realtime_update:
                    st.write("Double click to save crop")
                # Get a cropped image from the frontend
                cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                            aspect_ratio=aspect_ratio)

                # Manipulate cropped image at will
                st.write("Preview")
                cropped_img_1 = cropped_img.thumbnail((150,150))
                st.image(cropped_img)

    with col2:
        for _ in range(6):
            st.write('')
        st.write('  ---→')
        for _ in range(6):
            st.write('')

    with col3:
        st.write('학습')
        st.button('모델 학습')
        with st.expander('고급'):
            st.write('고급 기술')

    with col4:
        for _ in range(6):
            st.write('')
        st.write('  ---→')
        for _ in range(6):
            st.write('')

    with col5:
        col5_1, col5_2 = st.columns([2, 5])
        col5_1.write('미리보기')
        col5_2.button('모델 내보내기')

        st.write('여기서 모델을 미리 확인하려면 먼저 왼쪽에서 모델을 학습시켜야 합니다.')

elif option == '텍스트 분류':
    
    st.header('텍스트 분류')

    col1, col2 = st.columns(2)

    with col1:
        test = st.text_area('분류할 구문을 넣어주세요','')
        col3, col4 = st.columns(2)
        
        with col3:
            bt = st.button('test')
            if len(test) == 0 and bt:
                st.write('텍스트를 적어주세요')
            elif len(test) != 0 and bt:
                st.write('분류중입니다.')

    with col2:
        st.write('구문텍스트의 감정 분류')

elif option == 'Speech2Text':
    st.header('Speech2Text')

    col1, col2 = st.columns([2, 1])

    with col1:
        st.file_uploader('Upload a mp3 file')
    
    testfile = '' # 스피치로 받은 파일을 텍스트로 변환후 붙혀주기  
    if len(testfile) == 0:
        st.text_area('Speech to Text')
    else:
        st.text_area('Speech to Text')
        
elif option == '시계열 데이터 예측':
    st.header('시계열 데이터 예측')

    col1, col2 = st.columns([4, 6])

    with col1:  
        uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, type=['csv'])

        data_df = pd.read_csv('./data/AirPassengers.csv', encoding='cp949')
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.selectbox('날짜 데이터 선택',('Date1','Date2','Date3'))
            
        with col1_2:
            st.selectbox('예측할 데이터 선택',('Count1','Count2','Count3'))
             
    with col2:
        col1_1, col1_2, col1_3 = st.columns(3)
        col1_2.date_input('훈련 데이터 범위 지정', datetime.date(2015, 1, 1))
        col1_3.date_input('', datetime.date(2018, 12, 31))

        col2_1, col2_2, col2_3 = st.columns(3)
        col2_2.date_input('검증 데이터 범위 지정',datetime.date(2019, 1, 1))
        col2_3.date_input('', datetime.date(2022, 12, 31))

        col3_1, col3_2, col3_3 = st.columns(3)
        col3_2.date_input('예측 데이터 범위 지정',datetime.date(2019, 1, 1))
        col3_3.date_input('', datetime.date(2023, 1, 31))