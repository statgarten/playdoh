import streamlit as st
import image_classification as img_cls
import sentiment_analysis as sent_ans
import speech_to_text as stt
import time_series_forecasting as timeseries

def main():
    st.title("AI Applications")
    
    app_options = {
        "Image Classification": img_cls,
        "Sentiment Analysis": sent_ans,
        "Speech to Text": stt,
        "Predict Time Series": timeseries
    }
    
    app_choice = st.sidebar.selectbox("Choose an application", options=list(app_options.keys()))
    
    app_options[app_choice].main()

if __name__ == "__main__":
    main()


