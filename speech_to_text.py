import streamlit as st
import requests
import io

def main():
    st.header("Speech-to-Text with wav2vec")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "wma"])
    if uploaded_file is not None:
        bytes_data = io.BytesIO(uploaded_file.getvalue())
        audio_file = {"file": (uploaded_file.name, bytes_data, "audio/wav")}
        response = requests.post("http://localhost:8001/speech_to_text", files=audio_file)
        transcription = response.json()["transcription"]

        st.write("Transcription:")
        st.write(transcription)

# For running this file individually
# if __name__ == "__main__":
#     app()