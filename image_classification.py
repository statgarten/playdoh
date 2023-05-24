import streamlit as st
import pandas as pd
import altair as alt
import requests

from PIL import Image
from io import BytesIO

# 이미지 미리보기
def preview_img(uploaded_images, label_name, no):
    img = Image.open(uploaded_images[label_name][no])
    resize_img = img.resize((500, 500))
    # 여백을 위한 코드
    with st.container():
        for _ in range(3):
            st.write(' ')
    st.image(resize_img)
    st.caption('file name : ' + uploaded_images[label_name][no].name + ' / size : ' + str(uploaded_images[label_name][no].size / 1000) + 'KB (' + str(no + 1) + ' / ' + str(len(uploaded_images[label_name])) + ')')

# 각 이미지의 데이터셋 길이만큼 label생성
def label_create(data_set, label):
    
    labels = [label for _ in range(len(data_set))]
    
    return labels

# 버튼 모양 변경
def button_style():
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
                button[kind="primary"]:hover {
                    text-decoration: none;
                    color: black !important;
                }
                button[kind="primary"]:focus {
                    outline: none !important;
                    box-shadow: none !important;
                    color: black !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

def main():

    st.subheader("Image Classification")

    num_classes = st.sidebar.slider("Select number of classes", 2, 5) # set maximum class to 5

    # image upload
    with st.expander('Image Upload', expanded=True):
        uploaded_images = {}
        
        # 이미지 업로드 칸과 미리보기 공간 나누기
        empty, img_upload, empty, img_see, empty = st.columns([0.2, 4, 0.3, 6, 0.1])

        with img_upload:
            train_labels = []
            file_bytes_list = []              
            for i in range(num_classes):
                # labels = []
                # 이미지 label 이름 수정칸과 업로드 공간 분리
                img_label, empty, img_load = st.columns([2.0, 0.1, 6.5])    
                
                with img_label:
                    class_name = st.text_input(f"name of class {i+1}", f'class {i+1}')
                    train_labels.append(class_name)

                with img_load:
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

        # image preview
        with img_see:
            if len(list(uploaded_images.keys())) > 0:

                pick, no_input, empty = st.columns([6, 2, 2.4])

                with pick:
                    label_name = st.radio("Pick preview image list",
                                            list(uploaded_images.keys()),
                                            horizontal = True)

                empty ,prev, empty, img, empty, nex, empty = st.columns([0.2, 2, 0.5, 6, 0.5, 2, 0.2])

                # ← previous
                with prev:
                    for i in range(13):
                        with st.container():
                            st.empty()
                    if st.button('← previous', use_container_width = True):
                        st.session_state.lst_no -= 1

                # next →
                with nex:
                    for i in range(13):
                        with st.container():
                            st.empty()
                    if st.button('next →', use_container_width = True):
                        st.session_state.lst_no += 1

                # preview
                with img:
                    # 초기화
                    if 'lst_no' not in st.session_state:
                        st.session_state['lst_no'] = 0

                    # image slide
                    if st.session_state.lst_no >= len(uploaded_images[label_name]) - 1: 
                        st.session_state.lst_no = len(uploaded_images[label_name]) - 1
                        preview_img(uploaded_images, label_name, st.session_state['lst_no'])
                    elif st.session_state.lst_no < 1: 
                        st.session_state.lst_no = 0
                        preview_img(uploaded_images, label_name, st.session_state['lst_no'])                    
                    else:
                        preview_img(uploaded_images, label_name, st.session_state['lst_no'])

            # 아직 이미지를 업로드 안했을 때
            else:
                st.write('Pick preview image list')
                st.info('Please upload a train image')

    # model train
    with st.expander('Model Train', expanded=False):
        # Hyper parameter tuning
        empty, hyper_parameter_pick, empty, hyper_parameter_explanation, empty = st.columns([0.2, 2, 0.5, 6.5, 0.2]) 
        
        # HP_dict
        HP_dict = {
            'Learning rate' : 'Learning rate 란 \n한국에서 학습률이라고 불리는 Mahcine learning에서 training 되는 양 또는 단계를 의미합니다. \n\nLearning rate 기준 값 \nLearning rate(학습률)의 값을 어떻게 설정하느냐에 따라서 ML 결과가 달라집니다. 최적의 학습률을 설정해야지만 최종적으로 원하는 결과를 산출해낼 수 있습니다. Learning rate의 값이 적합하지 않을 경우, Overflow가 발생할 수도 있습니다. 한마디로 학습률이 너무 크면 Training 과정에서 발생하는 오류를 줄이지 못한다는 것입니다. 반면에 학습률이 너무 낮다고 해서 좋지만은 않습니다. 학습률이 너무 낮을 경우에는 ML 과정이 오래 걸리고 검증해내는 오류 값이 너무 많아져 Machine learning이 멈출 수가 있습니다. 한마디로 Learning rate가 높으면 산출되는 결과 속도가 빨라지지만 오류 값을 제대로 산출해내지 못하거나 오버플로우가 발생할 수 있고, 반대로 Learning rate가 너무 낮으면 산출되는 결과 속도가 느려지고 오류 값이 너무 많아져 실행 과정 자체가 멈출 수 있습니다. 따라서 적합한 Learning rate 값을 찾는 것이 중요합니다. \n\nLearning rate 초기값 \n일반적으로 0.1, 0.01, 0.001 등의 값을 시도해 볼 수 있습니다.',
            'Batch size': 'Batch size 란 \nBatch 크기는 모델 학습 중 parameter를 업데이트할 때 사용할 데이터 개수를 의미합니다. \n\nBatch size 예시 \n사람이 문제 풀이를 통해 학습해 나가는 과정을 예로 들어보겠습니다. Batch 크기는 몇 개의 문제를 한 번에 쭉 풀고 채점할지를 결정하는 것과 같습니다. 예를 들어, 총 100개의 문제가 있을 때, 20개씩 풀고 채점한다면 Batch 크기는 20입니다. 사람은 문제를 풀고 채점을 하면서 문제를 틀린 이유나 맞춘 원리를 학습합니다. 딥러닝 모델 역시 마찬가지입니다. Batch 크기만큼 데이터를 활용해 모델이 예측한 값과 실제 정답 간의 오차(conf. 손실함수)를 계산하여 Optimizer가 parameter를 업데이트합니다. \n\nBatch size 범위 \nBatch size가 너무 큰 경우 한 번에 처리해야 할 데이터의 양이 많아지므로, 학습 속도가 느려지고, 메모리 부족 문제가 발생할 위험이 있습니다. 반대로, Batch size가 너무 작은 경우 적은 데이터를 대상으로 가중치를 업데이트하고, 이 업데이트가 자주 발생하므로, 훈련이 불안정해집니다. ',
            'Epoch': 'Epoch 란 \n"에포크"라고 읽고 전체 데이터셋을 학습한 횟수를 의미합니다. \n\nEpoch 예시 \n사람이 문제집으로 공부하는 상황을 다시 예로 들어보겠습니다. epoch는 문제집에 있는 모든 문제를 처음부터 끝까지 풀고, 채점까지 마친 횟수를 의미합니다. 문제집 한 권 전체를 1번 푼 사람도 있고, 3번, 5번, 심지어 10번 푼 사람도 있습니다. epoch는 이처럼 문제집 한 권을 몇 회 풀었는지를 의미합니다. 즉 epoch가 10회라면, 학습 데이터 셋 A를 10회 모델에 학습시켰다는 것 입니다. \n\nEpoch 범위 \nEpoch를 높일수록, 다양한 무작위 가중치로 학습을 해보므로, 적합한 파라미터를 찾을 확률이 올라갑니다.(즉, 손실 값이 내려가게 됩니다.) 그러나, 지나치게 epoch를 높이게 되면, 그 학습 데이터셋에 과적합(Overfitting)되어 다른 데이터에 대해선 제대로 된 예측을 하지 못할 가능성이 올라갑니다.',
            'Optimizer': 'Optimizer 란 \n딥러닝 학습시 최대한 틀리지 않는 방향으로 학습해야 합니다. 얼마나 틀리는지(Loss)을 알게 하는 함수가 loss function(손실함수)입니다. loss function의 최솟값을 찾는 것을 학습 목표로 합니다. 최솟값을 찾아가는 과정이 최적화(Optimization), 이를 수행하는 알고리즘이 최적화 알고리즘(Optimizer)입니다. \n\n1. Adam \nAdagrad나 RMSProp처럼 각 파라미터마다 다른 크기의 업데이트를 진행하는 방법입니다. Adam의 직관은 local minima를 뛰어넘을 수 있다는 이유만으로 빨리 굴러가는 것이 아닌, minima의 탐색을 위해 조심스럽게 속도를 줄이고자 하는 것입니다. \n\n2. SGD \nSGD는 전체 입력 데이터로 가중치와 편향이 업데이트되는 것이 아니라, 그 안의 일부 데이터만 이용합니다. 전체 x, y 데이터에서 랜덤하게 배치 사이즈만큼 데이터를 추출하는데, 이를 미니 배치(mini batch)라고 합니다. 이를 통해 학습 속도를 빠르게 할 수 있을 뿐만 아니라 메모리도 절약할 수 있습니다. \n\n3. Adagrad \nAdagrad는 각 파라미터와 각 단계마다 학습률을 변경할 수 있습니다. second-order 최적화 알고리즘의 유형으로, 손실함수의 도함수에 대해 계산됩니다.'
        }
        with hyper_parameter_pick:
            with st.container():
                if st.button("**_Learning rate :question:_**", type='primary'):
                    st.session_state.explanation = HP_dict['Learning rate']
                learning_rate = st.text_input('', value = 0.0001, label_visibility='collapsed') 

            with st.container():
                if st.button("**_Batch size_** :question:", type='primary'):
                    st.session_state.explanation = HP_dict['Batch size']
                batch_size = st.text_input('', value = 20, label_visibility='collapsed')
                
            with st.container():
                if st.button("**_Epoch_** :question:", type='primary'):
                    st.session_state.explanation = HP_dict['Epoch']
                epoch = st.text_input('', value = 100, label_visibility='collapsed')

            with st.container():
                if st.button("**_Optimizer_** :question:", type='primary'):
                    st.session_state.explanation = HP_dict['Optimizer']
                opti = st.selectbox('', ('Adam', 'SGD', 'AdaGrad'), label_visibility='collapsed')

            # button style modify
            button_style()
          
        with hyper_parameter_explanation:
            if 'explanation' not in st.session_state:
                st.session_state['explanation'] = 'if you click hyper parameter name or ?, you can see hyper parameter explanation'
            
            # text_area의 font_color change
            st.markdown("""
                    <style>
                    .stTextArea [data-baseweb=base-input] [disabled=""]{
                        -webkit-text-fill-color: black;
                    }
                    </style>
                    """,unsafe_allow_html=True)

            st.text_area('**_Explanation_**', st.session_state['explanation'], height = 342, disabled = True)

        # Images to backend (/img_train)
        empty, train_txt, train_model = st.columns([12.8, 2.0, 2.0])

        # training model
        with train_model:
            training = False
            if st.button("Training Model"):
                if len(uploaded_images.keys()) >= 2:
                    training = True
                    # create labels
                    create_labels = []
                    for label_name in train_labels:
                        create_labels += label_create(uploaded_images[label_name], label_name)
                else:
                    st.markdown(':red[Please upload train image!!]')

        # print training text 
        with train_txt:        
            if training:
                with st.spinner('Training Model..'):
                    response = requests.post("http://localhost:8001/img_train", files=file_bytes_list, data={'labels':list(create_labels),
                                                                                                    'learning_rate':float(learning_rate),
                                                                                                    'batch_size':int(batch_size),
                                                                                                    'epoch':int(epoch),
                                                                                                    'opti':opti,
                                                                                                    'num_classes':num_classes})
                if response.ok:
                    st.success("Train completed : )")
                else:
                    st.error("Train error : (")
                    st.write(response)
                training = False
        
    # model test
    with st.expander('Model Test', expanded=False):

        test_img_upload, test_pred_prob, empty = st.columns([3.5, 6.5, 0.1])

        with test_img_upload:
            
            empty, test_img_load, empty = st.columns([0.7, 9, 0.5])

            with test_img_load:
                file_bytes_test_list = []

                # 1) Upload a test image and send the image to (POST)
                test_image = st.file_uploader("Upload a test image", type=['png', 'jpg', 'jpeg','tiff','webp'], accept_multiple_files=False)
                if test_image:

                    # png, jpg같이 채널이 차이나는 파일에 대한 채널수 동일 코드
                    test_file_byte= test_image.read()
                    test_image_tmp = Image.open(BytesIO(test_file_byte))
                    test_image_rgb = test_image_tmp.convert('RGB')
                    test_byte_arr = BytesIO()
                    test_image_rgb.save(test_byte_arr, format='JPEG')
                    file_byte_test = test_byte_arr.getvalue()

                    file_bytes_test_list.append(('files', BytesIO(file_byte_test)))
                    pred = requests.post("http://localhost:8001/img_test", files=file_bytes_test_list)

            empty, test_img, empty = st.columns([0.7, 8, 0.5])

            with test_img:
                if test_image:
                    img = Image.open(test_image)
                    resize_img = img.resize((450, 450))  
                    st.image(resize_img)
                    st.caption('file name : ' + test_image.name + ' / ' + 'size : ' + str(test_image.size / 1000) + 'KB')

        with test_pred_prob:
            if test_image:
                st.write('Prediction result')

                if pred.ok:
                    prediction = pred.json()

                    # prediction 결과를 담기 위한 DataFrame 생성
                    pred_df = pd.DataFrame(columns=['class_name', 'prediction_probability'])

                    for idx, pred_result in enumerate(prediction['prediction']):
                        pred_df.loc[idx] = [pred_result[0], round(pred_result[1] * 100, 2)]

                    with st.container():
                        st.write("해당 사진은 "+ str(round(prediction['prediction'][0][1] * 100, 2)) +"%의 확률로 '"+  prediction['prediction'][0][0] +"' 입니다.")

                    base = alt.Chart(pred_df, height=550).encode(x='prediction_probability',
                                                                 y='class_name:N',
                                                                 color=alt.Color('class_name:N', scale=alt.Scale(scheme='category10')),
                                                                 text='prediction_probability'
                                                                 )
                    # size=alt.value(30): 막대 굵기, align='left': , dx=10: 막대와의 거리, fontSize=20: 글자 사이즈
                    chart = base.mark_bar().encode(size=alt.value(30)) + base.mark_text(align='left', dx=10, fontSize=20)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.error('Please Train Model : <')
            else:
                st.write('Prediction result')
                st.info('Please upload a test image')

        # 2) Download the fine-tuned model (GET)
        empty, download_model = st.columns([8, 1.2])

        with download_model:
            if test_image and pred.ok:
                image_classification_model = requests.get('http://localhost:8001/model_download')
                st.download_button(label = 'Download Model',
                                   data = image_classification_model.content,
                                   file_name = 'image_classification_model.pth')
                
# For running this file individually
# if __name__ == "__main__":
#     app()