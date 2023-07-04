import streamlit as st
import requests
import io

def main():
    st.header("Speech-to-Text with wav2vec")
    left_column, right_column = st.columns(2)
    ll_column, lr_column, rl_column, rr_column   = st.columns(4)

    uploaded_file = left_column.file_uploader("Choose an audio file", type=["wav", "mp3", "wma"])
    submit_button = lr_column.button("Transcribe it!", use_container_width=True)

    if uploaded_file is not None:
        right_column.markdown('#')
        right_column.audio(uploaded_file)
        wav_data = io.BytesIO(uploaded_file.read())
        if submit_button:
                response = requests.post("http://localhost:8001/speech_to_text", files={"file": wav_data})
                if response.status_code == 200:
                    transcription = response.json()["transcription"]
                    st.write("Transcription:")
                    st.markdown(transcription[0])
                    st.download_button('Download', transcription[0], mime='text/plain')
                else:
                    st.error(response.status_code)

# For running this file individually
# if __name__ == "__main__":
#     app()