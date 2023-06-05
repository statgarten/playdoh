import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import requests
import json
import datetime
import io


# 원본 데이터와 예측 데이터를 가지고 라인그래프로 표현
def line_chart(data_df, int_col, pred_list, predict_plus_data, date):
    
    # line_chart 그리기
    line = pd.DataFrame({'raw':data_df[int_col],
                              'pred':np.NaN})
    
    # 그래프의 x축으로 잡기 위한 설정 
    line[date] = data_df[date]
    
    # 예측한 기간에만 예측 데이터 삽입
    line.loc[line.index >= len(line)-len(pred_list), 'pred'] = pred_list
    
    # 추가적으로 예측한 범위에 대해서 날짜를 하루씩 추가하여 데이터 삽입
    for idx, count in enumerate(predict_plus_data):
        pred_df = pd.DataFrame({'Date':[line[date][len(line)-1] + datetime.timedelta(days=1)],
                                'raw':[np.NaN],
                                'pred':[count]})
        line = pd.concat([line, pred_df], ignore_index = True)
    
    # # 그래프의 x축으로 잡기 위한 설정

    raw_layer = (alt.Chart(line, height=400)
        .encode(
            x="Date:T",
            y=alt.Y("raw:Q"),
        )
    )

    pred_layer = alt.Chart(line, height=400).encode(
            x="Date:T",
            y=alt.Y("pred:Q"),
        )
    
    chart = pred_layer.mark_line(color='red') + raw_layer.mark_line(color='blue')
    st.altair_chart(
        chart,
        use_container_width=True
    )
    # line = line.set_index(date)
    
    # # 그래프 생성
    # st.line_chart(line)

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

    global add_pred_list
    global pred_list

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

    explanation_title = translate('explanation_title', st.session_state.ko_en)
    explanation_text = translate('explanation_text', st.session_state.ko_en)

    training_validation_model_button = translate('training_validation_model_button', st.session_state.ko_en)
    prediction_model_button = translate('prediction_model_button', st.session_state.ko_en)

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
                    date = st.selectbox(decide_date_column,(train_df.select_dtypes(include=['object','datetime64','float64', 'int64']).columns.tolist()))
                    # date컬럼이 datetime으로 바뀌지 않으면
                    try :
                        train_df[date] = pd.to_datetime(train_df[date])
                    except:
                        st.caption(invalid_format)
                with pred_column:
                    int_col = st.selectbox(decide_pred_column,(train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()))

                ratio = int(len(train_df) * 0.8)
                
                with choice_arrange:
                    try:
                        _, train_s, _, train_e = st.columns([1, 3, 0.5, 3])
                        train_start = train_s.date_input(train_data_arrange, pd.to_datetime(train_df[date][0]))
                        train_end = train_e.date_input('',pd.to_datetime(train_df[date][ratio]), label_visibility='hidden')

                        _, val_s, _, val_e = st.columns([1, 3, 0.5, 3])
                        val_start = val_s.date_input(val_data_arrange, pd.to_datetime(train_df[date][ratio+1]))
                        val_end = val_e.date_input('', pd.to_datetime(train_df[date][len(train_df[date])-1]), label_visibility='hidden')

                        _, pred_s, _, pred_e = st.columns([1, 3, 0.5, 3])
                        pred_start = pred_s.date_input(pred_data_arrange, pd.to_datetime(train_df[date][ratio+1]))
                        pred_end = pred_e.date_input('', pd.to_datetime(train_df[date][len(train_df[date])-1]) + datetime.timedelta(days=30), label_visibility='hidden')
                        
                        train_start_index = train_df[train_df[date] == pd.to_datetime(train_start)].index[0]
                        train_end_index = train_df[train_df[date] == pd.to_datetime(train_end)].index[0] + 1
                        test_start_index = train_df[train_df[date] == pd.to_datetime(val_start)].index[0]
                        test_end_index = train_df[train_df[date] == pd.to_datetime(val_end)].index[0] + 1
                        pred_start_index = train_df[train_df[date] == pd.to_datetime(pred_start)].index[0]
                        
                        data_arrange = [train_start_index, train_end_index, test_start_index, test_end_index, pred_start_index, pred_end]
                    except:
                        _, date_column_not_exist = st.columns([0.5, 3.5])
                        with date_column_not_exist:
                            st.warning(date_column_not_exist_select_column)
            else:
                st.info(upload_info)
                

    with st.expander(expander_train_prediction, expanded=False):

        _, choice_hp, _, result, _ = st.columns([0.3, 3, 0.3, 7, 0.3])

        HP_dict = {
                'Window size' : '~',
                'Horizon factor': '~',
                'Epoch': 'Epoch 란 \n"에포크"라고 읽고 전체 데이터셋을 학습한 횟수를 의미합니다. \n\nEpoch 예시 \n사람이 문제집으로 공부하는 상황을 다시 예로 들어보겠습니다. epoch는 문제집에 있는 모든 문제를 처음부터 끝까지 풀고, 채점까지 마친 횟수를 의미합니다. 문제집 한 권 전체를 1번 푼 사람도 있고, 3번, 5번, 심지어 10번 푼 사람도 있습니다. epoch는 이처럼 문제집 한 권을 몇 회 풀었는지를 의미합니다. 즉 epoch가 10회라면, 학습 데이터 셋 A를 10회 모델에 학습시켰다는 것 입니다. \n\nEpoch 범위 \nEpoch를 높일수록, 다양한 무작위 가중치로 학습을 해보므로, 적합한 파라미터를 찾을 확률이 올라갑니다.(즉, 손실 값이 내려가게 됩니다.) 그러나, 지나치게 epoch를 높이게 되면, 그 학습 데이터셋에 과적합(Overfitting)되어 다른 데이터에 대해선 제대로 된 예측을 하지 못할 가능성이 올라갑니다.',
            }
        with choice_hp:
            with st.container():
                win_size, _, hori_fac = st.columns([1, 0.1, 1])
                with win_size:
                    if st.button(HP_window_size, type='primary'):
                        st.session_state.explanation = HP_dict['Window size']
                    window_size = st.text_input('', value = 20, label_visibility='collapsed') 
                with hori_fac:
                    if st.button(HP_horizon_factor, type='primary'):
                        st.session_state.explanation = HP_dict['Horizon factor']
                    horizon_factor = st.text_input('', value = 1, label_visibility='collapsed')
                
            with st.container():
                lear_rate, _, epo = st.columns([1,0.1,1])
                with lear_rate:
                    if st.button(HP_learning_rate, type='primary'):
                        st.session_state.explanation = HP_dict['learning_rate']
                    learning_rate = st.text_input('', value = 0.001, label_visibility='collapsed')
                with epo:
                    if st.button(HP_epoch, type='primary'):
                        st.session_state.explanation = HP_dict['Epoch']
                    epoch = st.text_input('', value = 1000, label_visibility='collapsed')

            st.text_area(explanation_title, explanation_text, height = 250, disabled = True)

        with result:
            if 'result' not in st.session_state:
                st.session_state['result'] = False
            
            if st.session_state['result'] and uploaded_file:
                st.session_state['result'] = False
                line_chart(train_df, int_col, pred_list, add_pred_list, date)
                st.write(st.session_state)
            else:
                st.text_area('', pred_graph, height = 502, disabled = True, label_visibility='collapsed')
                                    
            training_val, pred, download = st.columns(3)
            with training_val:
                if st.button(training_validation_model_button, use_container_width=True):
                    response = requests.post("http://localhost:8001/time_train_pred", files=csv_files, data={'data_arranges':list(data_arrange),
                                                                                                        'window_size':int(window_size),
                                                                                                        'horizon_factor':int(horizon_factor),
                                                                                                        'epoch':int(epoch),
                                                                                                        'learning_rate':float(learning_rate),
                                                                                                        'pred_col': int_col,
                                                                                                        'date': date})
                    if response.ok:
                        res = response.json()
                        pred_list = res['pred_list']
                        add_pred_list = res['pred_plus_list']
                    else:
                        st.write(response)

            with pred:
                if st.button(prediction_model_button, use_container_width=True):
                    st.session_state['result'] = True

            with download:
                st.button(model_download, use_container_width=True)

# For running this file individually
# if __name__ == "__main__":
#     app()