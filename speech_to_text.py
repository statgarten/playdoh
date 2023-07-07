import streamlit as st
import requests
import io
import json

def translate(key, language):
    with open(f'locale/stt_{language}.json', "r", encoding='utf-8') as file:
        translations = json.load(file)
        return translations[language][key]
    
def main():
    st.header(translate('sub_title', st.session_state.ko_en))

    left_column, right_column = st.columns(2)
    uploaded_file = left_column.file_uploader(translate('choose_audio', st.session_state.ko_en), type=["wav", "mp3", "flac", "ogg"])
    
    mode = st.radio(translate('radio', st.session_state.ko_en), (translate('online', st.session_state.ko_en), 
                                                                 translate('offline', st.session_state.ko_en)), horizontal=True)
    
    if mode == 'online' or mode == '온라인':

        api_addr, _, _, _  = st.columns(4)

        api_addr.markdown(
        f"<a style='display: block; text-align: left;' href=https://developers.vito.ai/docs/>Client ID and Secret link</a>",
        unsafe_allow_html=True,
        )

        client_id_input, client_secret_input, _, _  = st.columns(4)

        client_id = client_id_input.text_input(translate('Client_ID', st.session_state.ko_en),value="")
        client_secret = client_secret_input.text_input(translate('Client_Secret', st.session_state.ko_en),value="")

        ll_column, lr_column, rl_column, rr_column   = st.columns(4)
        submit_button = lr_column.button(translate('transcribe_button', st.session_state.ko_en), use_container_width=True)

        
        if uploaded_file is not None and len(client_id) > 0 and len(client_secret) > 0:
            st.session_state.transcription = None
            
            right_column.markdown('#')
            right_column.audio(uploaded_file)

            wav_data = io.BytesIO(uploaded_file.read())
            if submit_button:
                response = requests.post("http://localhost:8001/speech_to_text_api", files={"file": wav_data}, data={'client_id': client_id,
                                                                                                                     'client_secret': client_secret})
                if response.status_code != 200:
                    st.error(response.status_code)
                else:
                    transcription = response.json()["transcription"]
                    st.session_state.transcription = ''.join(transcription)
                    st.text_area(label = "Transcription:", value = st.session_state.transcription, disabled=True)
        _, right_column = st.columns(2)
        right_column.caption('<div style="text-align: right;">Model Api: https://developers.vito.ai/</div>', unsafe_allow_html=True)

    elif mode == 'offline' or mode == '오프라인':
        ll_column, lr_column, rl_column, rr_column   = st.columns(4)
        submit_button = lr_column.button(translate('transcribe_button', st.session_state.ko_en), use_container_width=True)
        if uploaded_file is not None:
            st.session_state.transcription = None

            right_column.markdown('#')
            right_column.audio(uploaded_file)

            wav_data = io.BytesIO(uploaded_file.read())
            if submit_button:
                response = requests.post("http://localhost:8001/speech_to_text", files={"file": wav_data})
                if response.status_code != 200:
                    st.error(response.status_code)
                else:
                    transcription = response.json()["transcription"]
                    st.session_state.transcription = ''.join(transcription)
                    st.text_area(label = "Transcription:", value = st.session_state.transcription, disabled=True)
                    _,_,_,_,_,rmr_column   = st.columns(6)
                    rmr_column.download_button(translate('download', st.session_state.ko_en), st.session_state.transcription, mime='text/plain')
        _, right_column = st.columns(2)
        right_column.caption('<div style="text-align: right;">Model Source: https://huggingface.co/facebook/wav2vec2-base-960h</div>', unsafe_allow_html=True)

# For running this file individually
# if __name__ == "__main__":
#     app()