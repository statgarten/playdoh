import streamlit as st
import image_classification as img_cls
import text_classification as txt_cls
import speech_to_text as stt
import predict_timeseries as timeseries

st.set_page_config(page_title="AI Applications", layout="wide")

def main():
    st.title("AI Applications")
    
    app_options = {
        "Image Classification": img_cls,
        "Text Classification": txt_cls,
        "Speech to Text": stt,
        "Predict Time Series": timeseries
    }
    
    app_choice = st.sidebar.selectbox("Choose an application", options=list(app_options.keys()))
    
    app_options[app_choice].main()

if __name__ == "__main__":
    main()


