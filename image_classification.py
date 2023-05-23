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
                img_label, empty, img_load = st.columns([1.8, 0.1, 7])    
                
                with img_label:
                    class_name = st.text_input(f"name of class {i+1}", f'class {i+1}')
                    train_labels.append(class_name)

                with img_load:
                    images = st.file_uploader('',accept_multiple_files=True, key=f'uploader{i}', type=['png', 'jpg', 'jpeg']) # 'tiff', 'webp'
                    if images:
                        uploaded_images[class_name] = [image for image in images]
                        for image, label in zip(images, uploaded_images[class_name]):
                            file_byte = image.read()
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
                st.info('Please Update a train image')

    # model train
    with st.expander('Model Train', expanded=False):
        # Hyper parameter tuning
        empty, hyper_parameter_pick, empty, hyper_parameter_explanation, empty = st.columns([0.2, 2, 0.5, 6.5, 0.2]) 

        # HP_dict
        HP_dict = {
            'Learning rate' : 'Learning rate는 한국에서 학습률이라고 불리는 Mahcine learning에서 training 되는 양 또는 단계를 의미한다. Learning rate(학습률)의 값을 어떻게 설정하느냐에 따라서 ML 결과가 달라진다. 최적의 학습률을 설정해야지만 최종적으로 원하는 결과를 산출해낼 수 있다. Learning rate의 값이 적합하지 않을 경우, Overflow가 발생할 수도 있다. 한마디로 학습률이 너무 크면 Training 과정에서 발생하는 오류를 줄이지 못한다는 것이다. 반면에 학습률이 너무 낮다고 해서 좋지만은 않다. 학습률이 너무 낮을 경우에는 ML 과정이 오래 걸리고 검증해내는 오류 값이 너무 많아져 Machine learning이 멈출 수가 있다. 한마디로 Learning rate가 높으면 산출되는 결과 속도가 빨라지지만 오류 값을 제대로 산출해내지 못하거나 오버플로우가 발생할 수 있고, 반대로 Learning rate가 너무 낮으면 산출되는 결과 속도가 느려지고 오류 값이 너무 많아져 실행 과정 자체가 멈출 수 있다. 따라서 적합한 Learning rate 값을 찾는 것이 중요하다. 일반적으로 0.1, 0.01, 0.001 등의 값을 시도해 볼 수 있습니다.',
            'Batch size': 'Description of Batch size',
            'Epoch': 'Description of Epoch',
            'Optimizer': 'Description of Optimizer'
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
            st.text_area('**_Explanation_**', st.session_state['explanation'], height = 342, disabled = True)

        # Images to backend (/img_train)
        empty, train_txt, train_model = st.columns([12.8, 2.0, 2.0])

        # training model
        with train_model:
            training = False
            if st.button("Training Model"):
                training = True
                # create labels
                create_labels = []
                for label_name in train_labels:
                    create_labels += label_create(uploaded_images[label_name], label_name)

        # print training text 
        with train_txt:        
            if training:
                with st.spinner('Training Model..'):
                    response = requests.post("http://localhost:8000/img_train", files=file_bytes_list, data={'labels':list(create_labels),
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
            
            empty, test_img_load, empty = st.columns([0.7, 9, 1])

            with test_img_load:
                file_bytes_test_list = []

                # 1) Upload a test image and send the image to (POST)
                test_image = st.file_uploader("Upload a test image", type=['png', 'jpg', 'jpeg','tiff','webp'])
                if test_image:
                    file_byte_test = test_image.read()
                    file_bytes_test_list.append(('files', BytesIO(file_byte_test)))
                    pred = requests.post("http://localhost:8000/img_test", files=file_bytes_test_list) # st.session_state['image_model'], , data={'model':st.session_state['image_model']}

            empty, test_img, empty = st.columns([0.7, 8, 1])

            with test_img:
                if test_image:
                    img = Image.open(test_image)
                    resize_img = img.resize((450, 450))  
                    st.image(resize_img)
                    st.caption('file name : ' + test_image.name + ' / ' + 'size : ' + str(test_image.size / 1000) + 'KB')

        with test_pred_prob:
            if test_image:
                st.write('Prediction Result')

                if pred.ok:
                    prediction = pred.json()

                    # prediction 결과를 담기 위한 DataFrame 생성
                    pred_df = pd.DataFrame(columns=['class', 'prediction_probability'])

                    for idx, pred_result in enumerate(prediction['prediction']):
                        pred_df.loc[idx] = [pred_result[0], round(pred_result[1] * 100, 2)]

                    with st.container():
                        st.write("해당 사진은 "+ str(round(prediction['prediction'][0][1] * 100, 2)) +"%의 확률로 '"+  prediction['prediction'][0][0] +"' 입니다.")

                    base = alt.Chart(pred_df, height=550).encode(x='prediction_probability',
                                                                 y='class:N',
                                                                 color=alt.Color('class:N', scale=alt.Scale(scheme='category10')),
                                                                 text='prediction_probability'
                                                                 )
                    # size=alt.value(30): 막대 굵기, align='left': , dx=10: 막대와의 거리, fontSize=20: 글자 사이즈
                    chart = base.mark_bar().encode(size=alt.value(30)) + base.mark_text(align='left', dx=10, fontSize=20)
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.error('Please Train Model : <')
            else:
                st.write('prediction result')
                st.info('Please Update a test image')

        # 2) Download the fine-tuned model (GET)
        empty, download_model = st.columns([8, 1.2])

        with download_model:
            if test_image and pred.ok:
                image_classification_model = requests.get('http://localhost:8000/model_download')
                st.download_button(label = 'Download Model',
                                   data = image_classification_model.content,
                                   file_name = 'image_classification_model.pth')
                
# For running this file individually
# if __name__ == "__main__":
#     app()