import streamlit as st
import requests
import io

def main():
    st.header("Speech-to-Text with wav2vec")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "wma"])
    if uploaded_file is not None:
        wav_data = io.BytesIO(uploaded_file.read())
        if wav_data is not None:
                response = requests.post("http://localhost:8001/speech_to_text", files={"file": wav_data})
                if response.status_code == 200:
                    st.success("good")
                    transcription = response.json()["transcription"]
                    st.write("Transcription:")
                    st.write(transcription)
                else:
                    st.error(response.status_code)

# For running this file individually
# if __name__ == "__main__":
#     app()