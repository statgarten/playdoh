import json
import streamlit as st
import image_classification as img_cls
import sentiment_analysis as sent_ans
import speech_to_text as stt
import time_series_forecasting as timeseries
from PIL import Image

def translate(key, language):
    with open(f'locale/main_{language}.json', "r", encoding='utf-8') as file:
        translations = json.load(file)
        return translations[language][key]

def main():
    st.set_page_config(layout="wide")

    logo = Image.open('./playdoh_logo.png')
    st.sidebar.image(logo, use_column_width=True)
 
    st.sidebar.markdown(
        f"<a style='display: block; text-align: center;' href=https://github.com/statgarten/playdoh>Github link</a>",
        unsafe_allow_html=True,
    )

    app_options = {
        "Image Classification": img_cls,
        "Sentiment Analysis": sent_ans,
        "Speech to Text": stt,
        "Time Series Forecasting": timeseries
    }

    app_choice = st.sidebar.selectbox("", options=list(app_options.keys()))

    # 기본이 english
    if 'ko_en' not in st.session_state:
        st.session_state['ko_en'] = 'en'

    with st.sidebar:
        en, ko = st.columns([1, 1])
        with en:
            if st.button('English', use_container_width=True) :
                st.session_state.ko_en = 'en'
        with ko:
            if st.button('한국어', use_container_width=True) :
                st.session_state.ko_en = 'ko'

    app_options[app_choice].main()

    # hide streamlit menu and footer
    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden;}
       footer {visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    
    # add customized footer
    st.sidebar.write("#")
    st.sidebar.write("#")
    st.sidebar.write("#")
    st.sidebar.write("#")
    st.sidebar.write("#")
    st.sidebar.write("#")
    st.sidebar.write("#")
    st.sidebar.write("#")
    st.sidebar.write("#")
    st.sidebar.write("#")

    text = translate('resolution', st.session_state.ko_en)
    st.sidebar.caption(f'<div style="text-align: center;">{text}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()