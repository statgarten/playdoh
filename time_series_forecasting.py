import streamlit as st
import pandas as pd
import requests
import json
import datetime
import io

# 각 언어에 따른 language json load
def load_translation(language):
    with open(f'locale/time_series_{language}.json', "r", encoding='utf-8') as file:
        translations = json.load(file)
    return translations

# 선택된 언어에 따라 번역된 텍스트를 반환합니다.
def translate(key, language):
    translations = load_translation(language)
    return translations[language][key]

def main():

    sub_title = translate('sub_title', st.session_state.ko_en)

    upload_train_csv = translate('upload_train_csv', st.session_state.ko_en)

    st.header(sub_title)
    
    with st.expander('모델 업로드 및 범위 지정', expanded=True):

        _, upload_csv, choice_arrange, _ = st.columns([0.3, 4, 6, 0.3])
        with upload_csv:

            uploaded_file = st.file_uploader(upload_train_csv, accept_multiple_files=False, type=['csv','excel','xlsx'])

            if uploaded_file: 

                # Read the file data
                file_data = uploaded_file.getvalue()

                # Create a file-like object
                file_obj = io.BytesIO(file_data)

                # Prepare the request payload
                files = {'file': file_obj}

                train_df = pd.read_csv(uploaded_file, encoding='cp949')
                data_column, pred_column = st.columns(2)
                with data_column:
                    date = st.selectbox('날짜 데이터를 선택해주세요',(train_df.select_dtypes(include=['object','datetime64']).columns.tolist()))
                    # date컬럼이 datetime으로 바뀌지 않으면
                    try :
                        train_df[date] = pd.to_datetime(train_df[date])
                    except:
                        st.caption('잘못된 형식입니다.')
                with pred_column:
                    int_col = st.selectbox('예측할 데이터를 선택해주세요',(train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()))

                ratio = int(len(train_df) * 0.8)
                
                with choice_arrange:
                    _, train_s, _, train_e = st.columns([1, 3, 0.5, 3])
                    train_start = train_s.date_input('훈련 데이터 범위 지정',pd.to_datetime(train_df[date][0]))
                    train_end = train_e.date_input('',pd.to_datetime(train_df[date][ratio]))

                    _, val_s, _, val_e = st.columns([1, 3, 0.5, 3])
                    val_start = val_s.date_input('검증 데이터 범위 지정',pd.to_datetime(train_df[date][ratio+1]))
                    val_end = val_e.date_input('', pd.to_datetime(train_df[date][len(train_df[date])-1]))
                    
                    _, pred_s, _, pred_e = st.columns([1, 3, 0.5, 3])
                    pred_start = pred_s.date_input('예측 데이터 범위 지정',pd.to_datetime(train_df[date][ratio+1]))
                    pred_end = pred_e.date_input('', pd.to_datetime(train_df[date][len(train_df[date])-1]) + datetime.timedelta(days=30))
            else:
                st.info('업로드시 컬럼 선택과 범위 지정이 가능합니다.')

    with st.expander('모델 학습 및 예측', expanded=False):

        _, choice_hp, _, result, _ = st.columns([0.3, 3, 0.3, 7, 0.3])

        HP_dict = {
                'Window size' : '~',
                'Horizon factor': '~',
                'Epoch': 'Epoch 란 \n"에포크"라고 읽고 전체 데이터셋을 학습한 횟수를 의미합니다. \n\nEpoch 예시 \n사람이 문제집으로 공부하는 상황을 다시 예로 들어보겠습니다. epoch는 문제집에 있는 모든 문제를 처음부터 끝까지 풀고, 채점까지 마친 횟수를 의미합니다. 문제집 한 권 전체를 1번 푼 사람도 있고, 3번, 5번, 심지어 10번 푼 사람도 있습니다. epoch는 이처럼 문제집 한 권을 몇 회 풀었는지를 의미합니다. 즉 epoch가 10회라면, 학습 데이터 셋 A를 10회 모델에 학습시켰다는 것 입니다. \n\nEpoch 범위 \nEpoch를 높일수록, 다양한 무작위 가중치로 학습을 해보므로, 적합한 파라미터를 찾을 확률이 올라갑니다.(즉, 손실 값이 내려가게 됩니다.) 그러나, 지나치게 epoch를 높이게 되면, 그 학습 데이터셋에 과적합(Overfitting)되어 다른 데이터에 대해선 제대로 된 예측을 하지 못할 가능성이 올라갑니다.',
            }
        with choice_hp:
            with st.container():
                if st.button('Window size', type='primary'):
                    st.session_state.explanation = HP_dict['Window size']
                window_size = st.text_input('', value = 20, label_visibility='collapsed') 

            with st.container():
                if st.button('Horizon factor', type='primary'):
                    st.session_state.explanation = HP_dict['Horizon factor']
                horizon_factor = st.text_input('', value = 1, label_visibility='collapsed')
                
            with st.container():
                if st.button('Epoch', type='primary'):
                    st.session_state.explanation = HP_dict['Epoch']
                epoch = st.text_input('', value = 100, label_visibility='collapsed')

            st.text_area('explanation','asd', height = 200, disabled = True)

        with result:
            if 'result' not in st.session_state:
                st.session_state['result'] = '학습 및 예측 진행 후 결과가 출력됩니다.'

            st.text_area('',st.session_state['result'], height = 502, disabled = True, label_visibility='collapsed')

            training_val, pred, download = st.columns(3)
            with training_val:
                if st.button('학습 및 검증', use_container_width=True):
                    response = requests.post("http://localhost:8001/time_train", files=files)
                    if response.ok:
                        res = response.json()
                        st.session_state['result'] = res['message']
                    else:
                        st.write(response)

            with pred:
                st.button('예측', use_container_width=True)

            with download:
                st.button('모델 다운로드', use_container_width=True)




# For running this file individually
# if __name__ == "__main__":
#     app()