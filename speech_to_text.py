import streamlit as st
import requests

def main():
    st.header("Speech-to-Text with wav2vec")
    
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
        response = requests.post("http://localhost:8000/transcribe", files=files)
        transcription = response.json()["transcription"]

        st.write("Transcription:")
        st.write(transcription)

# For running this file individually
# if __name__ == "__main__":
#     app()