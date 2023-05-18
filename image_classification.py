import streamlit as st
import requests
import numpy as np
from PIL import Image

# @st.cache_resource
# def create_list():
#     l = [0, 1]
#     return l

# 이미지 미리보기
def preview_img(uploaded_images, label_name, no):
    img = Image.open(uploaded_images[label_name][no])
    resize_img = img.resize((500, 500))         
    st.subheader(uploaded_images[label_name][no].name)
    st.image(resize_img)

def main():
    st.subheader("Image Classification")

    num_classes = st.sidebar.slider("Select number of classes", 2, 5) # set maximum class to 5

    # image upload
    with st.expander('Image Upload', expanded=True):
        uploaded_images = {}
        
        # 이미지 업로드 칸과 미리보기 공간 나누기
        img_upload, empty, img_see, empty = st.columns([4, 0.3, 6, 0.1])

        with img_upload:
            
            for i in range(num_classes):
                # 이미지 label 이름 수정칸과 업로드 공간 분리
                img_label, empty, img_load = st.columns([2, 0.5, 7])    

                with img_label:
                    class_name = st.text_input(f"name of class {i+1}", f'class {i+1}')

                with img_load:
                    images = st.file_uploader('',accept_multiple_files=True, key=f'uploader{i}', type=['png', 'jpg', 'jpeg']) # 'tiff', 'webp'
                    if images:
                        uploaded_images[class_name] = [image for image in images]
        
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

            else:
                st.write('Pick preview image list')

    # model train
    with st.expander('Model Train', expanded=False):
        # Hyper parameter tuning
        hyper_parameter_pick, empty, hyper_parameter_explanation = st.columns([2.5, 0.5, 6.5]) 

        with hyper_parameter_pick:
            with st.container():
                st.markdown("**_Learning rate_** :question:")
                learning_rate = float(st.text_input('', value = 0.0001, label_visibility='collapsed'))
                st.empty()    

            with st.container():
                st.markdown("**_Batch size_** :question:")
                batch_size = int(st.text_input('', value = 20, label_visibility='collapsed'))
                st.empty()

            with st.container():
                st.markdown("**_Epoch_** :question:")
                epoch = int(st.text_input('', value = 100, label_visibility='collapsed'))
                st.empty()

            with st.container():
                st.markdown("**_Optimizer_** :question:")
                opti = st.selectbox('',
                                    ('Adam', 'Adam2', 'Adam3'), label_visibility='collapsed')
                st.empty()

            st.write(learning_rate, batch_size, epoch, opti)

        with hyper_parameter_explanation:
            st.text_area('**_Explanation_**', 'asd', height = 342, disabled = True)

        # Images to backend (/img_train)
        if st.button("Train Model"):
            files = [(f"{class_name}_{i}", image) for class_name, images in uploaded_images.items() for i, image in enumerate(images)]
            response = requests.post("http://localhost:8000/img_train", files=files)
            if response.ok:
                st.success("Train completed :)")
            else:
                st.error("Train error :(")

    # model test
    with st.expander('Model Test', expanded=False):
        # TODO:  
        # 1) Upload a test image and send the image to (POST)
        test_image = st.file_uploader("Upload a test image", type=['png', 'jpg', 'jpeg'])

        # 2) Download the fine-tuned model (GET)
        if st.button("Download Model"):
            return True


# For running this file individually
# if __name__ == "__main__":
#     app()