import streamlit as st
import pandas as pd
import altair as alt
import requests
import json

from PIL import Image
from io import BytesIO

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
    response = requests.post("http://localhost:8001/img_train", files=file_bytes_list, data={'labels':list(create_labels),
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

    pred = requests.post("http://localhost:8001/img_test", files=file_bytes_test_list)

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
def css_style():
    st.markdown("""
                <style>
                button[kind="primary"] {
                    background: none!important;
                    border: none;
                    padding: 0!important;
                    color: black !important;
                    text-decoration: none;
                    cursor: pointer;
                    border: none !important;
                }
            
                [data-testid="stFileUploadDropzone"] div div::before {content:"Upload your image"}
                [data-testid="stFileUploadDropzone"] div div span{display:none;}

                .stTextArea [data-baseweb=base-input] [disabled=""]{
                -webkit-text-fill-color: black;}

                .css-1li7dat {
                    display: none;
                }
      
                </style>
                """,
                unsafe_allow_html=True,
            )

def main():
    # css styling
    css_style()

    # st.session_state['explanation'] 초기화
    explanation_session_clear()

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
            'Learning rate' : 'Learning rate 란 \n한국에서 학습률이라고 불리는 Mahcine learning에서 training 되는 양 또는 단계를 의미합니다. \n\nLearning rate 기준 값 \nLearning rate(학습률)의 값을 어떻게 설정하느냐에 따라서 ML 결과가 달라집니다. 최적의 학습률을 설정해야지만 최종적으로 원하는 결과를 산출해낼 수 있습니다. Learning rate의 값이 적합하지 않을 경우, Overflow가 발생할 수도 있습니다. 한마디로 학습률이 너무 크면 Training 과정에서 발생하는 오류를 줄이지 못한다는 것입니다. 반면에 학습률이 너무 낮다고 해서 좋지만은 않습니다. 학습률이 너무 낮을 경우에는 ML 과정이 오래 걸리고 검증해내는 오류 값이 너무 많아져 Machine learning이 멈출 수가 있습니다. 한마디로 Learning rate가 높으면 산출되는 결과 속도가 빨라지지만 오류 값을 제대로 산출해내지 못하거나 오버플로우가 발생할 수 있고, 반대로 Learning rate가 너무 낮으면 산출되는 결과 속도가 느려지고 오류 값이 너무 많아져 실행 과정 자체가 멈출 수 있습니다. 따라서 적합한 Learning rate 값을 찾는 것이 중요합니다. \n\nLearning rate 초기값 \n일반적으로 0.1, 0.01, 0.001 등의 값을 시도해 볼 수 있습니다.',
            'Batch size': 'Batch size 란 \nBatch 크기는 모델 학습 중 parameter를 업데이트할 때 사용할 데이터 개수를 의미합니다. \n\nBatch size 예시 \n사람이 문제 풀이를 통해 학습해 나가는 과정을 예로 들어보겠습니다. Batch 크기는 몇 개의 문제를 한 번에 쭉 풀고 채점할지를 결정하는 것과 같습니다. 예를 들어, 총 100개의 문제가 있을 때, 20개씩 풀고 채점한다면 Batch 크기는 20입니다. 사람은 문제를 풀고 채점을 하면서 문제를 틀린 이유나 맞춘 원리를 학습합니다. 딥러닝 모델 역시 마찬가지입니다. Batch 크기만큼 데이터를 활용해 모델이 예측한 값과 실제 정답 간의 오차(conf. 손실함수)를 계산하여 Optimizer가 parameter를 업데이트합니다. \n\nBatch size 범위 \nBatch size가 너무 큰 경우 한 번에 처리해야 할 데이터의 양이 많아지므로, 학습 속도가 느려지고, 메모리 부족 문제가 발생할 위험이 있습니다. 반대로, Batch size가 너무 작은 경우 적은 데이터를 대상으로 가중치를 업데이트하고, 이 업데이트가 자주 발생하므로, 훈련이 불안정해집니다. ',
            'Epoch': 'Epoch 란 \n"에포크"라고 읽고 전체 데이터셋을 학습한 횟수를 의미합니다. \n\nEpoch 예시 \n사람이 문제집으로 공부하는 상황을 다시 예로 들어보겠습니다. epoch는 문제집에 있는 모든 문제를 처음부터 끝까지 풀고, 채점까지 마친 횟수를 의미합니다. 문제집 한 권 전체를 1번 푼 사람도 있고, 3번, 5번, 심지어 10번 푼 사람도 있습니다. epoch는 이처럼 문제집 한 권을 몇 회 풀었는지를 의미합니다. 즉 epoch가 10회라면, 학습 데이터 셋 A를 10회 모델에 학습시켰다는 것 입니다. \n\nEpoch 범위 \nEpoch를 높일수록, 다양한 무작위 가중치로 학습을 해보므로, 적합한 파라미터를 찾을 확률이 올라갑니다.(즉, 손실 값이 내려가게 됩니다.) 그러나, 지나치게 epoch를 높이게 되면, 그 학습 데이터셋에 과적합(Overfitting)되어 다른 데이터에 대해선 제대로 된 예측을 하지 못할 가능성이 올라갑니다.',
            'Optimizer': 'Optimizer 란 \n딥러닝 학습시 최대한 틀리지 않는 방향으로 학습해야 합니다. 얼마나 틀리는지(Loss)을 알게 하는 함수가 loss function(손실함수)입니다. loss function의 최솟값을 찾는 것을 학습 목표로 합니다. 최솟값을 찾아가는 과정이 최적화(Optimization), 이를 수행하는 알고리즘이 최적화 알고리즘(Optimizer)입니다. \n\nOpimizer 종류 \n1. Adam \nAdagrad나 RMSProp처럼 각 파라미터마다 다른 크기의 업데이트를 진행하는 방법입니다. Adam의 직관은 local minima를 뛰어넘을 수 있다는 이유만으로 빨리 굴러가는 것이 아닌, minima의 탐색을 위해 조심스럽게 속도를 줄이고자 하는 것입니다. \n\n2. SGD \nSGD는 전체 입력 데이터로 가중치와 편향이 업데이트되는 것이 아니라, 그 안의 일부 데이터만 이용합니다. 전체 x, y 데이터에서 랜덤하게 배치 사이즈만큼 데이터를 추출하는데, 이를 미니 배치(mini batch)라고 합니다. 이를 통해 학습 속도를 빠르게 할 수 있을 뿐만 아니라 메모리도 절약할 수 있습니다. \n\n3. Adagrad \nAdagrad는 각 파라미터와 각 단계마다 학습률을 변경할 수 있습니다. second-order 최적화 알고리즘의 유형으로, 손실함수의 도함수에 대해 계산됩니다.'
        }
        with hyper_parameter_pick:
            with st.container():
                if st.button(HP_learning_rate, type='primary'):
                    st.session_state.explanation = HP_dict['Learning rate']
                learning_rate = st.text_input('', value = 0.0001, label_visibility='collapsed') 

            with st.container():
                if st.button(HP_batch_size, type='primary'):
                    st.session_state.explanation = HP_dict['Batch size']
                batch_size = st.text_input('', value = 20, label_visibility='collapsed')
                
            with st.container():
                if st.button(HP_epoch, type='primary'):
                    st.session_state.explanation = HP_dict['Epoch']
                epoch = st.text_input('', value = 100, label_visibility='collapsed')

            with st.container():
                if st.button(HP_optimizer, type='primary'):
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
                        st.balloons() # good
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
                image_classification_model = requests.get('http://localhost:8001/model_download')
                st.download_button(label = model_download,
                                data = image_classification_model.content,
                                file_name = 'image_classification_model.pth')
    
# For running this file individually
# if __name__ == "__main__":
#     app()