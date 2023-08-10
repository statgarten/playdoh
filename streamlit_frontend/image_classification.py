import streamlit as st
from streamlit_modal import Modal
import pandas as pd
import altair as alt
import requests
import json

from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
import os

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")

# 학습 이미지 미리보기
def train_preview_img(uploaded_images, label_name, no):
    img = Image.open(uploaded_images[label_name][no])
    resize_img = img.resize((500, 500))
    # 여백을 위한 코드
    with st.container():
        for _ in range(3):
            st.write(' ')
    st.image(resize_img)
    st.caption('file name : ' + uploaded_images[label_name][no].name + ' / size : ' + str(uploaded_images[label_name][no].size / 1000) + 'KB (' + str(no + 1) + ' / ' + str(len(uploaded_images[label_name])) + ')')

# 테스트 이미지 미리보기
def test_preview_img(test_image):
    img = Image.open(test_image)
    resize_img = img.resize((450, 450))  
    st.image(resize_img)
    st.caption('file name : ' + test_image.name + ' / ' + 'size : ' + str(test_image.size / 1000) + 'KB')

# 각 이미지의 데이터셋 길이만큼 label생성
def label_create(data_set, label):
    labels = [label for _ in range(len(data_set))]
    return labels

# add class
def class_count_plus():
    if st.session_state['num_classes'] >= 5:
        st.session_state['num_classes'] = 5
    else:
        st.session_state['num_classes'] += 1

# delete class
def class_count_minus():
    if st.session_state['num_classes'] <= 2:
        st.session_state['num_classes'] = 2
    else:
        st.session_state['num_classes'] -= 1

# 사진 전체 삭제
def clear_img(num_class):
    for i in range(num_class):
        del st.session_state[f'uploader{i}']

# 설명 세션 초기화
def explanation_session_clear():
    if 'explanation' in st.session_state:
        del st.session_state['explanation']

# 학습 이미지 업로드
# @st.cache(allow_output_mutation=True)
def train_img_upload(i, uploaded_images, class_name, file_bytes_list):

    images = st.file_uploader(' ',accept_multiple_files=True, key=f'uploader{i}', type=['png', 'jpg', 'jpeg', 'tiff','webp'],label_visibility='hidden') # 'tiff', 'webp'
    if images:
        uploaded_images[class_name] = [image for image in images]
        for image, _ in zip(images, uploaded_images[class_name]):
            # png, jpg같이 채널이 차이나는 파일에 대한 채널수 동일 코드
            file_byte = image.read()
            image_tmp = Image.open(BytesIO(file_byte))
            image_rgb = image_tmp.convert('RGB')
            byte_arr = BytesIO()
            image_rgb.save(byte_arr, format='JPEG')
            file_byte = byte_arr.getvalue()
            file_bytes_list.append(('files', BytesIO(file_byte)))
    
    return file_bytes_list


# prev, nex에 empty 생성
def create_empty():
    for i in range(13):
        with st.container():
            st.empty()

# 이미지 슬라이드
def image_slide(uploaded_images, label_name):
    # 초기화
    if 'lst_no' not in st.session_state:
        st.session_state['lst_no'] = 0
    # image slide
    if st.session_state.lst_no >= len(uploaded_images[label_name]) - 1: 
        st.session_state.lst_no = len(uploaded_images[label_name]) - 1
        train_preview_img(uploaded_images, label_name, st.session_state['lst_no'])
    elif st.session_state.lst_no < 1: 
        st.session_state.lst_no = 0
        train_preview_img(uploaded_images, label_name, st.session_state['lst_no'])                    
    else:
        train_preview_img(uploaded_images, label_name, st.session_state['lst_no'])

# create_labels
def create_label(train_labels, uploaded_images):
    create_labels = []
    for label_name in train_labels:
        create_labels += label_create(uploaded_images[label_name], label_name)
    return create_labels

# train request
def train_request(file_bytes_list, create_labels, learning_rate, batch_size, epoch, opti, num_classes):
    response = requests.post(f"{BACKEND_URL}/img_train", files=file_bytes_list, data={'labels':list(create_labels),
                                                                                            'learning_rate':float(learning_rate),
                                                                                            'batch_size':int(batch_size),
                                                                                            'epoch':int(epoch),
                                                                                            'opti':opti,
                                                                                            'num_classes':num_classes})
    return response

# upload test image and request
# png, jpg같이 채널이 차이나는 파일에 대한 채널수 동일 코드
def test_image_upload_request(test_image):
    file_bytes_test_list = []

    test_file_byte= test_image.read()
    test_image_tmp = Image.open(BytesIO(test_file_byte))
    test_image_rgb = test_image_tmp.convert('RGB')
    test_byte_arr = BytesIO()
    test_image_rgb.save(test_byte_arr, format='JPEG')
    file_byte_test = test_byte_arr.getvalue()
    file_bytes_test_list.append(('files', BytesIO(file_byte_test)))

    pred = requests.post(f"{BACKEND_URL}/img_test", files=file_bytes_test_list)

    return pred

# prediction_result
def prediction_result(pred):
    prediction = pred.json()
    # prediction 결과를 담기 위한 DataFrame 생성
    pred_df = pd.DataFrame(columns=['class_name', 'prediction_probability'])

    for idx, pred_result in enumerate(prediction['prediction']):
        pred_df.loc[idx] = [pred_result[0], round(pred_result[1] * 100, 2)]

    _, pred_result = st.columns([3, 7])

    with pred_result:
        with st.container():
            st.markdown("###### 해당 사진은 __"+ str(round(prediction['prediction'][0][1] * 100, 2)) +"__%의 확률로 '__"+  prediction['prediction'][0][0] +"__' 입니다.")

    base = alt.Chart(pred_df, height=550).encode(x='prediction_probability',
                                                y='class_name:N',
                                                color=alt.Color('class_name:N', scale=alt.Scale(scheme='category10')),
                                                text='prediction_probability'
                                                )
    # size=alt.value(30): 막대 굵기, align='left': , dx=10: 막대와의 거리, fontSize=20: 글자 사이즈
    chart = base.mark_bar().encode(size=alt.value(30)) + base.mark_text(align='left', dx=10, fontSize=20)
    st.altair_chart(chart, use_container_width=True)

# 각 언어에 따른 language json load
def load_translation(language):
    with open(f'locale/image_{language}.json', "r", encoding='utf-8') as file:
        translations = json.load(file)
    return translations

# 선택된 언어에 따라 번역된 텍스트를 반환합니다.
def translate(key, language):
    translations = load_translation(language)
    return translations[language][key]

# css 조정
# button 모양 조정
# fileupload text 조정
# text_area font-color 조정
# text_input text 조정
#     button[kind="primary"]:hover { 
#     text-decoration: none;
#     color: black !important;
# }
# button[kind="primary"]:focus {
#     outline: none !important;
#     box-shadow: none !important;
#     color: black !important;
# }  
# 
# .stTextArea [data-baseweb=base-input] [disabled=""]{
# -webkit-text-fill-color: black;}
# .css-upb2o1 .stButton .e1ewe7hr10
def css_style():
    st.markdown("""
                <style>
                .block-container div:nth-child(4) > ul > li div:nth-child(2) div.streamlit-expanderContent > div:nth-child(1) > div > div:nth-child(1) .stButton button{
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
                }

                button[kind="secondary"]:hover { 
                    text-decoration: none;
                }

                [data-testid="stFileUploadDropzone"] div div span{display:none;}
                </style>
                """,
                unsafe_allow_html=True
            )

def main():

    # css styling
    css_style()

    # st.session_state['explanation'] 초기화
    explanation_session_clear()

    # 언어에 따른 업로드 문구 변경
    # if st.session_state.ko_en == 'en':
    #     st.markdown("""
    #         <style>
    #         [data-testid="stFileUploadDropzone"] div div::before {content:"Upload your image"}
    #         </style>
    #     """,unsafe_allow_html=True)
    # else:
    #     st.markdown("""
    #         <style>
    #         [data-testid="stFileUploadDropzone"] div div::before {content:"사진을 업로드 해주세요"}
    #         </style>
    #     """,unsafe_allow_html=True)

    # st.session_state['num_classes'] 초기화
    if 'num_classes' not in st.session_state:
        st.session_state['num_classes'] = 2

    # 해당 언어의 json load후 번역
    sub_title = translate('sub_title', st.session_state.ko_en)

    expander_image = translate('expander_image', st.session_state.ko_en)
    expander_train = translate('expander_train', st.session_state.ko_en)
    expander_test = translate('expander_test', st.session_state.ko_en)

    add_class_button = translate('add_class_button', st.session_state.ko_en)
    del_class_button = translate('del_class_button', st.session_state.ko_en)

    train_image_preview = translate('train_image_preview', st.session_state.ko_en)
    train_image_prev = translate('train_image_prev', st.session_state.ko_en)
    train_image_next = translate('train_image_next', st.session_state.ko_en)

    upload_train_image = translate('upload_train_image', st.session_state.ko_en)
    upload_test_image = translate('upload_test_image', st.session_state.ko_en)

    explanation_title = translate('explanation_title', st.session_state.ko_en)
    explanation_text = translate('explanation_text', st.session_state.ko_en)

    HP_learning_rate = translate('HP_learning_rate', st.session_state.ko_en)
    HP_batch_size = translate('HP_batch_size', st.session_state.ko_en)
    HP_epoch = translate('HP_epoch', st.session_state.ko_en)
    HP_optimizer = translate('HP_optimizer', st.session_state.ko_en)

    ex_learning_rate = translate('ex_learning_rate', st.session_state.ko_en)
    ex_batch_size = translate('ex_batch_size', st.session_state.ko_en)
    ex_epoch = translate('ex_epoch', st.session_state.ko_en)
    ex_optimizer = translate('ex_optimizer', st.session_state.ko_en)

    training_model_button = translate('training_model_button', st.session_state.ko_en)
    training_model_spinner = translate('training_model_spinner', st.session_state.ko_en)
    training_model_complete = translate('training_model_complete', st.session_state.ko_en)
    training_model_error = translate('training_model_error', st.session_state.ko_en)

    prediction_result_title = translate('prediction_result', st.session_state.ko_en)
    please_train_model = translate('please_train_model', st.session_state.ko_en)
    model_download = translate('model_download', st.session_state.ko_en)

    st.subheader(sub_title)

    # image upload
    with st.expander(expander_image, expanded=True):

        uploaded_images = {}
        # 이미지 업로드 칸과 미리보기 공간 나누기
        _, img_upload, _, img_see, _ = st.columns([0.2, 4, 0.3, 6, 0.1])
        
        with img_upload:

            train_labels = []
            file_bytes_list = [] 

            add_class, del_class = st.columns([1, 1])
            with add_class:
                st.button(add_class_button, on_click=class_count_plus, use_container_width=True)
            with del_class:
                st.button(del_class_button, on_click=class_count_minus, use_container_width=True)
            # with clear_all_img:
            #     if st.button('Claer_all_img', use_container_width=True):
            #         clear_img(st.session_state['num_classes'])
            
            for i in range(st.session_state.num_classes):
                # labels = []
                # 이미지 label 이름 수정칸과 업로드 공간 분리
                img_label, _, img_load = st.columns([2.0, 0.1, 6.5])    
                
                with img_label:
                    class_name = translate('class_name', st.session_state.ko_en) + f'{i + 1}'
                    class_num = translate('class_num', st.session_state.ko_en) + f'{i + 1}'
                    class_name = st.text_input(class_name, class_num)
                    train_labels.append(class_name)
                        
                with img_load:
                    # with st.form(f'my_form{i}', clear_on_submit=True):
                    # train image upload
                    file_bytes_list = train_img_upload(i, uploaded_images, class_name, file_bytes_list)
            # st.write(st.session_state) # uploader0 / uploader1
                        # clear_button = st.form_submit_button(label="Clear")
                        # if clear_button:
                        #     st.session_state.pop(f'my_form{i}', None)

        # image preview
        with img_see:
            if len(list(uploaded_images.keys())) > 0:
                pick, _ = st.columns([6, 4.4])
                with pick:
                    label_name = st.radio(train_image_preview,
                                            list(uploaded_images.keys()),
                                            horizontal = True)
                _ ,prev, _, img, _, nex, _ = st.columns([0.2, 2, 0.5, 6, 0.5, 2, 0.2])
                # ← previous
                with prev:
                    # 위치 조정을 위한 empty
                    create_empty()
                    if st.button(train_image_prev, use_container_width = True):
                        st.session_state.lst_no -= 1
                # next →
                with nex:
                    # 위치 조정을 위한 empty
                    create_empty()
                    if st.button(train_image_next, use_container_width = True):
                        st.session_state.lst_no += 1
                # preview
                with img:
                    # 미리보기 사진 슬라이드
                    image_slide(uploaded_images, label_name)
            # 아직 이미지를 업로드 안했을 때
            else:
                st.write(train_image_preview)
                st.info(upload_train_image)

    # model train
    with st.expander(expander_train, expanded=False):
        # Hyper parameter tuning
        _, hyper_parameter_pick, _, hyper_parameter_explanation, _ = st.columns([0.2, 2, 0.5, 6.5, 0.2]) 
        
        # HP_dict
        HP_dict = {
            'Learning rate' : ex_learning_rate,
            'Batch size': ex_batch_size,
            'Epoch': ex_epoch,
            'Optimizer': ex_optimizer 
        }
        with hyper_parameter_pick:
            with st.container():
                if st.button(HP_learning_rate):
                    st.session_state.explanation = HP_dict['Learning rate']
                learning_rate = st.text_input('', value = 0.0001, label_visibility='collapsed') 

            with st.container():
                if st.button(HP_batch_size):
                    st.session_state.explanation = HP_dict['Batch size']
                batch_size = st.text_input('', value = 20, label_visibility='collapsed')
                
            with st.container():
                if st.button(HP_epoch):
                    st.session_state.explanation = HP_dict['Epoch']
                epoch = st.text_input('', value = 100, label_visibility='collapsed')

            with st.container():
                if st.button(HP_optimizer):
                    st.session_state.explanation = HP_dict['Optimizer']
                opti = st.selectbox('', ('Adam', 'SGD', 'AdaGrad'), label_visibility='collapsed')

        with hyper_parameter_explanation:
            if 'explanation' not in st.session_state:
                st.session_state['explanation'] = explanation_text
            st.text_area(explanation_title, st.session_state['explanation'], height = 342, disabled = True)

        # Images to backend (/img_train)
        _, train_txt, train_model, _ = st.columns([12.8, 2.7, 2.0, 0.4])

        # training model
        with train_model:
            training = False
            if st.button(training_model_button, use_container_width=True):
                training = True
    
        # print training text 
        with train_txt:        
            if training :
                if len(uploaded_images.keys()) !=0 and len(uploaded_images.keys()) >= st.session_state.num_classes:
                    # create labels
                    create_labels = create_label(train_labels, uploaded_images)

                    with st.spinner(training_model_spinner):
                            # 학습 요청
                        response = train_request(file_bytes_list, create_labels, learning_rate, batch_size, epoch, opti, st.session_state.num_classes)
                    if response.ok: 
                        st.success(training_model_complete)
                        # st.balloons()
                    else:
                        st.error(training_model_error)
                        st.write(response)
                    training = False
                else:
                    # st.markdown(':red[Please upload train image!!]')
                    st.warning(upload_train_image)
                    training = False

    # model test
    with st.expander(expander_test, expanded=False):

        test_img_upload, test_pred_prob, _ = st.columns([3.5, 6.5, 0.1])

        with test_img_upload:
            
            _, test_img_load, _ = st.columns([0.7, 9, 0.5])

            with test_img_load:
                # 1) Upload a test image and send the image to (POST)
                test_image = st.file_uploader(upload_test_image, type=['png', 'jpg', 'jpeg','tiff','webp'], accept_multiple_files=False)
                if test_image:
                    # 테스트 사진에 대한 upload 및 request
                    pred = test_image_upload_request(test_image)
                    
            _, test_img, _ = st.columns([0.7, 8, 0.5])

            with test_img:
                if test_image:
                    # 테스트 사진 미리보기
                    test_preview_img(test_image)

        with test_pred_prob:
            if test_image:
                st.write(prediction_result_title)
                if pred.ok:
                    # 예측 그래프 함수
                    prediction_result(pred)
                else:
                    st.error(please_train_model)
            else:
                st.write(prediction_result_title)
                st.info(upload_test_image)

        # 2) Download the fine-tuned model (GET)
        _, download_model = st.columns([8, 1.2])
        with download_model:
            if test_image and pred.ok:
                image_classification_model = requests.get(f'{BACKEND_URL}/img_classification_model_download')
                st.download_button(label = model_download,
                                data = image_classification_model.content,
                                file_name = 'image_classification_model.pth')
    
    _, right_column = st.columns(2)
    right_column.caption(f'<div style="text-align: right;">Model Source: https://huggingface.co/google/mobilenet_v1_0.75_192</div>', unsafe_allow_html=True)
    
# For running this file individually
# if __name__ == "__main__":
#     app()