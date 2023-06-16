import streamlit as st
import image_classification as img_cls
import sentiment_analysis as sent_ans
import speech_to_text as stt
import time_series_forecasting as timeseries

def main():

    st.set_page_config(layout="wide")
 
    st.title("Welcome to Playdoh!")
    
    app_options = {
        "Image Classification": img_cls,
        "Sentiment Analysis": sent_ans,
        "Speech to Text": stt,
        "Time Series Forecasting": timeseries
    }
    
    app_choice = st.sidebar.selectbox("Choose an application", options=list(app_options.keys()))

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

if __name__ == "__main__":
    main()