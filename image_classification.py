import streamlit as st
import requests

def main():
    st.title("Image Classification")

    num_classes = st.sidebar.slider("Select number of classes", 2, 5) # set maximum class to 5

    uploaded_images = {}
    for i in range(num_classes):
        class_name = st.text_input(f"Enter the name of class {i+1}")
        images = st.file_uploader(f"Upload image for class {class_name}", accept_multiple_files=True, key=f'uploader{i}', type=['png', 'jpg', 'jpeg'])
        if images:
            uploaded_images[class_name] = [image for image in images]

    # Images to backend (/img_train)
    if st.button("Train Model"):
        files = [(f"{class_name}_{i}", image) for class_name, images in uploaded_images.items() for i, image in enumerate(images)]
        response = requests.post("http://localhost:8000/img_train", files=files)
        if response.ok:
            st.success("Train completed :)")
        else:
            st.error("Train error :(")

    # TODO:  
    # 1) Upload a test image and send the image to (POST)
    test_image = st.file_uploader("Upload a test image", type=['png', 'jpg', 'jpeg'])

    # 2) Download the fine-tuned model (GET)
    if st.button("Download Model"):
        return True


# For running this file individually
# if __name__ == "__main__":
#     app()