import streamlit as st
import requests
import numpy as np
import pickle

from PIL import Image
from io import BytesIO
# @st.cache_resource
# def create_list():
#     l = [0, 1]
#     return l

# 이미지 미리보기
def preview_img(uploaded_images, label_name, no):
    img = Image.open(uploaded_images[label_name][no])
    resize_img = img.resize((500, 500))         
    st.image(resize_img)
    st.caption('file name : ' + uploaded_images[label_name][no].name)
    st.caption('file size : ' + str(uploaded_images[label_name][no].size / 1000) + 'MB')

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
        img_upload, empty, img_see, empty = st.columns([4, 0.3, 6, 0.1])

        with img_upload:
            train_labels = []
            file_bytes_list = []              
            for i in range(num_classes):
                # labels = []
                # 이미지 label 이름 수정칸과 업로드 공간 분리
                img_label, empty, img_load = st.columns([2, 0.5, 7])    
                
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

                # with no_input:
                #     st.session_state.lst_no = st.number_input('Insert image number',
                #                                               min_value=0,
                #                                               max_value=len(uploaded_images[label_name])-1,
                #                                               step = 1)
                # st.write(st.session_state.lst_no)

                prev, empty, img, empty, nex = st.columns([2, 0.5, 6, 0.5, 2])

                # ← previous
                with prev:
                    for i in range(13):
                        with st.container():
                            st.empty()
                    if st.button('← previous', use_container_width = True):
                        st.session_state.lst_no -= 1

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

                # next →
                with nex:
                    for i in range(13):
                        with st.container():
                            st.empty()
                    if st.button('next →', use_container_width = True):
                        st.session_state.lst_no += 1

            # 아직 이미지를 업로드 안했을 때
            else:
                st.write('Pick preview image list')

    # model train
    with st.expander('Model Train', expanded=False):
        # Hyper parameter tuning
        hyper_parameter_pick, empty, hyper_parameter_explanation = st.columns([2.5, 0.5, 6.5]) 

        # HP_dict
        HP_dict = {
            'Learning rate' : 'Description of Learning rate',
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
                opti = st.selectbox('', ('Adam', 'Adam2', 'Adam3'), label_visibility='collapsed')

            # button style modify
            button_style()
       
        with hyper_parameter_explanation:
            if 'explanation' not in st.session_state:
                st.session_state['explanation'] = 'hyper parameter explanation'
            st.text_area('**_Explanation_**', st.session_state['explanation'], height = 342, disabled = True)

        # Images to backend (/img_train)
        empty, train_txt, train_model = st.columns([8, 1.2, 0.8])

        with train_txt:
            if 'train_txt' not in st.session_state:
                st.session_state['train_txt'] = ''

        with train_model:
            
            if st.button("Train Model", use_container_width=True):
                st.session_state['train_txt'] = 'Training Model'
                
                # create labels
                create_labels = []
                for label_name in train_labels:
                    create_labels += label_create(uploaded_images[label_name], label_name)
                # post image_list, label_list
                # 
                response = requests.post("http://localhost:8001/img_train", files=file_bytes_list, data={'labels':list(create_labels),
                                                                                                'learning_rate':float(learning_rate),
                                                                                                'batch_size':int(batch_size),
                                                                                                'epoch':int(epoch),
                                                                                                'opti':opti,
                                                                                                'num_classes':num_classes})
                if response.ok:
                    st.success("Train completed b")
                    res = response.json()
                    model = response.content
                    # serialized_model = pickle.dumps(model)
                    st.session_state['image_model'] = model
                    st.session_state['model'] = model
                    # st.write(res['Model'])
                    # st.write(model)
                    # st.write(res['Encoder'])
                else:
                    st.error("Train error p")
                    st.write(response)

        # st.write(float(learning_rate), int(batch_size), int(epoch), opti)

    # model test
    with st.expander('Model Test', expanded=False):

        test_img_upload, test_pred = st.columns([3, 7])

        with test_img_upload:
            
            empty, test_img_load, empty = st.columns([0.5, 9, 1])

            with test_img_load:
                file_bytes_test_list = []
                # TODO:  
                # 1) Upload a test image and send the image to (POST)
                test_image = st.file_uploader("Upload a test image", type=['png', 'jpg', 'jpeg'])
                if test_image:
                    file_byte = test_image.read()
                    file_bytes_test_list.append(('files', BytesIO(file_byte)))

                    pred = requests.post("http://localhost:8001/img_test", files=file_bytes_test_list, data= {'model':st.session_state['model']}) # st.session_state['image_model'], , data={'model':st.session_state['image_model']}

            empty, test_img, empty = st.columns([1, 8, 1])

            with test_img:
                if test_image:
                    img = Image.open(test_image)
                    resize_img = img.resize((500, 500))  
                    st.image(resize_img)
                    st.caption('test_file_name : ' + test_image.name)
                    st.caption('test_file_size : ' + str(test_image.size / 1000) + 'MB') 

        with test_pred:
            if test_image:
                st.write('prediction result')
                if pred.ok:
                    res = pred.json()
                    # st.write(res['Model'])
                    pred_labels = res['pred_label']
                    pred_prob = round(res['prob'] * 100, 2)
                    st.subheader(f'이 사진은 {pred_labels}입니다. (probability: {pred_prob}%)')
                else:
                    st.write('error')
                    st.subheader(pred)
            else:
                st.write('prediction result')

        # 2) Download the fine-tuned model (GET)
        if st.button("Download Model"):
            return True


# For running this file individually
# if __name__ == "__main__":
#     app()