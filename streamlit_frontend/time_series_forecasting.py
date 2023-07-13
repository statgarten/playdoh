import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import requests
import json
import datetime
import io

from dotenv import load_dotenv
import os

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

# 원본 데이터와 예측 데이터를 가지고 라인그래프로 표현
def line_chart(data_df, int_col, pred_list, predict_plus_data, date):
    
    # line_chart 그리기
    line = pd.DataFrame({'raw':data_df[int_col],
                         'test':np.NaN,
                         'pred':np.NaN})
    
    # 그래프의 x축으로 잡기 위한 설정 
    line[date] = data_df[date]
    
    # 예측한 기간에만 예측 데이터 삽입
    line.loc[line.index >= len(line)-len(pred_list), 'test'] = pred_list
    
    # 추가적으로 예측한 범위에 대해서 날짜를 하루씩 추가하여 데이터 삽입
    for _, count in enumerate(predict_plus_data):
        pred_df = pd.DataFrame({date:[line[date][len(line)-1] + datetime.timedelta(days=1)],
                                'raw':[np.NaN],
                                'test':[np.NaN],
                                'pred':[count]})
        line = pd.concat([line, pred_df], ignore_index = True)

    raw_layer = (alt.Chart(line, height=450)
        .encode(
            x=date + ":T",
            y=alt.Y("raw:Q" , title='value')
        )
    )

    test_layer = (alt.Chart(line, height=450)
        .encode(
            x=date + ":T",
            y=alt.Y("test:Q", title='value'),
        )
    )
    
    pred_layer = (alt.Chart(line, height=450)
        .encode(
            x=date + ":T",
            y=alt.Y("pred:Q", title='value'),
        )
    )

    chart = test_layer.mark_line(color='red') + raw_layer.mark_line(color='blue') + pred_layer.mark_line(color='green')
    st.altair_chart(
        chart,
        use_container_width=True
    )

# 각 언어에 따른 language json load
def load_translation(language):
    with open(f'locale/time_series_{language}.json', "r", encoding='utf-8') as file:
        translations = json.load(file)
    return translations

# 선택된 언어에 따라 번역된 텍스트를 반환합니다.
def translate(key, language):
    translations = load_translation(language)
    return translations[language][key]

# 설명 세션 초기화
def explanation_session_clear():
    if 'explanation' in st.session_state:
        del st.session_state['explanation']

# .block-container div:nth-child(5) > ul > li div.streamlit-expanderContent > div  > div > div div:nth-child(2) > div:nth-child(1) > div .stButton button{
#                     background: none!important;
#                     border: none;
#                     padding: 0!important;
#                     text-decoration: none;
#                     cursor: pointer;
#                     border: none !important;
#                 }

# CSS 수정
def css_style():
    st.markdown("""
        <style>
               
            .block-container div:nth-child(4) > ul > li > div:nth-child(2) > div > div:nth-child(1) > div > div > div > div:nth-child(1) > div > div:nth-child(1) > div > div > div:nth-child(1) > div:nth-child(1) > div > div:nth-child(1) > div > button{
                    background: none!important;
                    border: none;
                    padding: 0!important;
                    text-decoration: none;
                    cursor: pointer;
                    border: none !important;
                }

            .block-container div:nth-child(4) > ul > li > div:nth-child(2) > div > div:nth-child(1) > div > div > div > div:nth-child(1) > div > div:nth-child(2) > div > div > div:nth-child(1) > div:nth-child(1) > div > div:nth-child(1) > div > button{
                    background: none!important;
                    border: none;
                    padding: 0!important;
                    text-decoration: none;
                    cursor: pointer;
                    border: none !important;
                }

            .block-container div:nth-child(4) > ul > li > div:nth-child(2) > div > div:nth-child(1) > div > div > div > div:nth-child(1) > div > div:nth-child(1) > div > div > div:nth-child(3) > div:nth-child(1) > div > div:nth-child(1) > div > button{
                    background: none!important;
                    border: none;
                    padding: 0!important;
                    text-decoration: none;
                    cursor: pointer;
                    border: none !important;
                }

            .block-container div:nth-child(4) > ul > li > div:nth-child(2) > div > div:nth-child(1) > div > div > div > div:nth-child(1) > div > div:nth-child(2) > div > div > div:nth-child(3) > div:nth-child(1) > div > div:nth-child(1) > div > button{
                    background: none!important;
                    border: none;
                    padding: 0!important;
                    text-decoration: none;
                    cursor: pointer;
                    border: none !important;
                }

            button[kind="secondary"]:focus {
                outline: none !important;
                box-shadow: none !important;
                color: !important;
                }

            button[kind="secondary"]:hover { 
                text-decoration: none;
                color: !important;
            }

        </style>
        
        """, unsafe_allow_html=True)

def main():

    # CSS 수정
    css_style()

    # 설명 세션 초기화
    explanation_session_clear()

    global add_pred_list
    global pred_list
    global test_x_tensor
    global prev_window_size 
    global result_req
    global response

    sub_title = translate('sub_title', st.session_state.ko_en)
    
    expander_upload_arrange = translate('expander_upload_arrange', st.session_state.ko_en)
    
    upload_train_csv = translate('upload_train_csv', st.session_state.ko_en)

    decide_date_column = translate('decide_date_column', st.session_state.ko_en)
    decide_pred_column = translate('decide_pred_column', st.session_state.ko_en)

    invalid_format = translate('invalid_format', st.session_state.ko_en)

    train_data_arrange = translate('train_data_arrange', st.session_state.ko_en)
    val_data_arrange = translate('val_data_arrange', st.session_state.ko_en)
    pred_data_arrange = translate('pred_data_arrange', st.session_state.ko_en)

    date_column_not_exist_select_column = translate('date_column_not_exist_select_column', st.session_state.ko_en)

    upload_info = translate('upload_info', st.session_state.ko_en)

    expander_train_prediction = translate('expander_train_prediction', st.session_state.ko_en)

    HP_learning_rate = translate('HP_learning_rate', st.session_state.ko_en)
    HP_epoch = translate('HP_epoch', st.session_state.ko_en)
    HP_window_size = translate('HP_window_size', st.session_state.ko_en)
    HP_horizon_factor = translate('HP_horizon_factor', st.session_state.ko_en)

    ex_learning_rate = translate('ex_learning_rate', st.session_state.ko_en)
    ex_epoch = translate('ex_epoch', st.session_state.ko_en)
    ex_window_size = translate('ex_window_size', st.session_state.ko_en)
    ex_horizon_factor = translate('ex_horizon_factor', st.session_state.ko_en)

    explanation_title = translate('explanation_title', st.session_state.ko_en)
    explanation_text = translate('explanation_text', st.session_state.ko_en)

    training_validation_model_button = translate('training_validation_model_button', st.session_state.ko_en)
    training_model_spinner = translate('training_model_spinner', st.session_state.ko_en)
    training_model_complete = translate('training_model_complete', st.session_state.ko_en)

    prediction_model_button = translate('prediction_model_button', st.session_state.ko_en)

    after_train = translate('after_train', st.session_state.ko_en)

    time_series_forecasting = translate('time_series_forecasting', st.session_state.ko_en)

    pred_after_down = translate('pred_after_down', st.session_state.ko_en)

    pred_graph = translate('pred_graph', st.session_state.ko_en)

    model_download = translate('model_download', st.session_state.ko_en)

    st.header(sub_title)
    
    with st.expander(expander_upload_arrange, expanded=True):

        _, upload_csv, choice_arrange, _ = st.columns([0.3, 4, 6, 0.3])
        with upload_csv:

            uploaded_file = st.file_uploader(upload_train_csv, accept_multiple_files=False, type=['csv','excel','xlsx'])

            if uploaded_file: 

                # Read the file data
                csv_file_data = uploaded_file.getvalue()
                # Create a file-like object
                csv_file_obj = io.BytesIO(csv_file_data)
                # Prepare the request payload
                csv_files = {'file': csv_file_obj}
                if uploaded_file.name.split('.')[-1] == 'csv':
                    train_df = pd.read_csv(uploaded_file, encoding='utf-8')
                else:
                    train_df = pd.read_excel(uploaded_file)
                date_column, pred_column = st.columns(2)
                with date_column:
                    try :
                        date = st.selectbox(decide_date_column,(train_df.select_dtypes(include=['object','datetime64']).columns.tolist()))
                        # date컬럼이 datetime으로 바뀌지 않으면
                        train_df[date] = pd.to_datetime(train_df[date])
                    except:
                        st.caption(invalid_format)
                with pred_column:
                    int_col = st.selectbox(decide_pred_column,(train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()))

                ratio = int(len(train_df) * 0.7)
                
                with choice_arrange:
                    try:
                        _, train_s, _, train_e = st.columns([1, 3, 0.5, 3])
                        train_start = train_s.date_input(train_data_arrange, pd.to_datetime(train_df[date][0]))
                        train_end = train_e.date_input('',pd.to_datetime(train_df[date][ratio]), label_visibility='hidden')

                        train_start_index = train_df[train_df[date] == pd.to_datetime(train_start)].index[0]
                        train_end_index = train_df[train_df[date] == pd.to_datetime(train_end)].index[0] + 1
                    
                        _, test_text, _ = st.columns([1, 5.6, 1])  
                        with test_text:
                            st.caption(val_data_arrange)

                        _, val_s, _, val_e = st.columns([1, 3, 0.5, 3])
                        val_start = val_s.date_input('val_start', pd.to_datetime(train_df[date][train_end_index]), label_visibility='collapsed', disabled=True)
                        val_end = val_e.date_input('val_end', pd.to_datetime(train_df[date][len(train_df[date])-1]), label_visibility='collapsed', disabled=True)

                        test_start_index = train_df[train_df[date] == pd.to_datetime(val_start)].index[0]
                        test_end_index = train_df[train_df[date] == pd.to_datetime(val_end)].index[0] + 1
                        # test date의 10%를 예측
                        num_features = (test_end_index - test_start_index) // 10
                        
                        pred_e_day = (val_end-val_start) // datetime.timedelta(days=10)

                        _, pred_text, _ = st.columns([1, 5.6, 1])  

                        with pred_text:
                            st.caption(pred_data_arrange)

                        _, pred_s, _, pred_e = st.columns([1, 3, 0.5, 3])
                    
                        pred_start = pred_s.date_input('pred_start', pd.to_datetime(train_df[date][len(train_df[date])-1]), label_visibility='collapsed', disabled=True)  
                        pred_end = pred_e.date_input('pred_end', pd.to_datetime(train_df[date][len(train_df[date])-1]) + datetime.timedelta(days=pred_e_day), label_visibility='collapsed', disabled=True)
                        
                        pred_start_index = train_df[train_df[date] == pd.to_datetime(pred_start)].index[0]
                        data_arrange = [train_start_index, train_end_index, test_start_index, test_end_index, pred_start_index, pred_end]
                    except:
                        _, date_column_not_exist = st.columns([0.5, 3.5])
                        with date_column_not_exist:
                            st.warning(date_column_not_exist_select_column)
                    
            else:
                st.info(upload_info)
                
    with st.expander(expander_train_prediction, expanded=False):

        _, choice_hp, _, pred_result, _ = st.columns([0.3, 3, 0.3, 7, 0.3])

        HP_dict = {
                'Window size' : ex_window_size,
                'Horizon factor': ex_horizon_factor,
                'Epoch': ex_epoch,
                'Learning_rate' : ex_learning_rate
            }
        
        with choice_hp:
            with st.container():
                win_size, _, hori_fac = st.columns([1, 0.1, 1])
                with win_size:
                    if st.button(HP_window_size):
                        st.session_state.explanation = HP_dict['Window size']
                    window_size = st.text_input('', value = 30, label_visibility='collapsed') 
                with hori_fac:
                    if st.button(HP_horizon_factor):
                        st.session_state.explanation = HP_dict['Horizon factor']
                    horizon_factor = st.text_input('', value = 1, label_visibility='collapsed')
                
            with st.container():
                lear_rate, _, epo = st.columns([1,0.1,1])
                with lear_rate:
                    if st.button(HP_learning_rate):
                        st.session_state.explanation = HP_dict['Learning_rate']
                    learning_rate = st.text_input('', value = 0.0001, label_visibility='collapsed')
                with epo:
                    if st.button(HP_epoch):
                        st.session_state.explanation = HP_dict['Epoch']
                    epoch = st.text_input('', value = 200, label_visibility='collapsed')

            if 'explanation' not in st.session_state:
                st.session_state['explanation'] = explanation_text
            st.text_area(explanation_title, st.session_state.explanation, height = 250, disabled = True)

            train_test, prediction = st.columns(2)

            with train_test:
                if uploaded_file:
                    if train_start < train_end: # 학습의 범위에서 시작이 끝보다 크게 설정 되었을 때
                        if st.button(training_validation_model_button, use_container_width=True) and uploaded_file:
                            prev_window_size = int(window_size)
                            with st.spinner(training_model_spinner):
                                response = requests.post(f"{BACKEND_URL}/time_train", files=csv_files, data={'data_arranges':list(data_arrange),
                                                                                                                    'window_size':int(window_size),
                                                                                                                    'horizon_factor':int(horizon_factor),
                                                                                                                    'epoch':int(epoch),
                                                                                                                    'learning_rate':float(learning_rate),
                                                                                                                    'pred_col': int_col,
                                                                                                                    'date': date})
                                
                                if response.ok:
                                    res = response.json()
                                    if res['data_success']:
                                        test_x_tensor = res['test_x_tensor']
                                        st.write(training_model_complete)
                                    else:
                                        # window_size > scaled_test 인 경우 진행 불가
                                        st.write('윈도우 사이즈를 ' + str(res['scaled_size']) + '보다 줄여서 학습을 진행해 주세요')
                                else:
                                    st.write(response)

                        # prediction이 안에 있는 이유는 잘못된 학습 범위나 파일을 업로드하지 않았을 때 한번에 처리하기 위함 
                        with prediction:
                            if uploaded_file and st.button(prediction_model_button, use_container_width=True):
                                try:
                                    if prev_window_size == int(window_size): 
                                        result_req = requests.post(f"{BACKEND_URL}/time_pred", data={'test_x_tensor':list(test_x_tensor),
                                                                                                            'num_features':num_features,
                                                                                                            'window_size':int(window_size)})
                                        if result_req.ok:
                                            resu = result_req.json()
                                            pred_list = resu['pred_list']
                                            add_pred_list = resu['predict_additional_list']
                                        else:
                                            st.write(result_req)
                                    else:
                                        st.write(after_train)
                                except:
                                    st.write(after_train)    
                    else:
                        st.caption('학습 범위를 제대로 설정해주세요')
                else:
                    st.caption(upload_train_csv)

        with pred_result:
            try:
                st.write(time_series_forecasting)
                if result_req.ok:
                    line_chart(train_df, int_col, pred_list, add_pred_list, date)
            except:
                st.text_area(' ', pred_graph, height = 454, disabled = True, label_visibility='collapsed')

            _, _, download = st.columns([4, 4, 2.1])
            with download:
                try:
                    if uploaded_file and result_req.ok:
                        time_series_model = requests.get(f'{BACKEND_URL}/time_series_model_download')       
                        st.download_button(label = model_download,
                                        data = time_series_model.content,
                                        file_name = 'time_series_forecasting_model.pth',
                                        use_container_width=True)
                except:
                    st.caption(pred_after_down)
                    
# For running this file individually
# if __name__ == "__main__":
#     app()