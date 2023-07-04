import streamlit as st
import requests
import io

def main():
    st.header("Speech-to-Text with wav2vec")
    left_column, right_column = st.columns(2) 

    uploaded_file = left_column.file_uploader("Choose an audio file", type=["wav", "mp3", "wma"])

    if uploaded_file is not None:
        right_column.video(uploaded_file)
        wav_data = io.BytesIO(uploaded_file.read())
        if wav_data is not None:
                response = requests.post("http://localhost:8001/speech_to_text", files={"file": wav_data})
                if response.status_code == 200:
                    transcription = response.json()["transcription"]
                    left_column.write("Transcription:")
                    left_column.write(transcription[0])
                else:
                    left_column.error(response.status_code)

# For running this file individually
# if __name__ == "__main__":
#     app()