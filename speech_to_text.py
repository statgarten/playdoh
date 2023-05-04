import streamlit as st
from function.functions import load_model, preprocess_audio, convert_audio_to_text

MODEL_ID = "facebook/wav2vec2-base"
SAMPLE_FILE = "sample.wav"

def main():
    st.title("Speech to Text")
    st.write("Upload a WAV file to get the transcript")
    
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file)
        audio_input = preprocess_audio(uploaded_file)
        
        st.write("Loading model...")
        processor, model = load_model(MODEL_ID)
        
        with st.spinner("Transcribing..."):
            transcript = convert_audio_to_text(audio_input, processor, model)
        
        st.write(f"Transcript: {transcript}")