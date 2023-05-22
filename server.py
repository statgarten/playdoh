import numpy as np
import pickle

import torch.nn.init
import torch.nn as nn
import torch.cuda as tc
import torch.optim as optim # 최적화 함수
import torchvision.transforms as transforms # 전처리
import torchvision.models as models # 사전 학습 모델

from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder

from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from io import BytesIO, BufferedReader

app = FastAPI()
# server내 공유
model = None
encoder = None
device = None

# LabelEncdoing
def label_encoding(labels):
    
    encoder = LabelEncoder()
    encoder.fit(np.array(labels))
    train_labels = encoder.transform(np.array(labels))
    
    return encoder, train_labels

# preprcessing image
transform_img = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[R채널 평균, G채널 평균, B채널 평균] , std=[R채널 표준편차, G채널 표준편차, B채널 표준편차])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ImageNet이 학습한 수백만장의 이미지의 RGB 각각의 채널에 대한 평균은 0.485, 0.456, 0.406 그리고 표준편차는 0.229, 0.224, 0.225
    
])

# create model
def create_model(num_classes, device):
    
    model = models.mobilenet_v3_large(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
          
    # 마지막 레이어의 입력 채널 수를 가져옴
    last_channel = model.classifier[-1].in_features

    # 새로운 선형 레이어를 생성하여 마지막 레이어를 대체
    model.classifier[-1] = nn.Linear(last_channel, num_classes)
    
    return model.to(device)

# trian model
def train_model(learning_rate, num_epochs, dataloader, device, model):

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            # Images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
    return model

# images: dict[str, list[UploadFile]] = File(...)
# labels:list[str], 
@app.post("/img_train")
async def train_model_endpoint(labels:list[str], 
                               learning_rate: float = Form(...), 
                               batch_size: int = Form(...),
                               epoch: int = Form(...),
                               opti: str = Form(...), 
                               num_classes: int = Form(...),
                               files: list[UploadFile] = File(...)):
    global encoder
    # encode label
    encoder, train_labels = label_encoding(labels)
    encoder = encoder

    # set gpu or cpu 
    device = 'cuda' if tc.is_available() else 'cpu'

    # create dataset
    # 1. Image transform
    dataset = []
    for file, label in zip(files, train_labels):
        image = Image.open(BytesIO(await file.read()))
        image = transform_img(image)
        dataset.append((image, label))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)    
    
    global model

    # 2. Create the model 
    model = create_model(num_classes, device)
    
    # 3. Train the model with HP
    model = train_model(learning_rate, epoch, dataloader, device, model)
    print(type(model))

    return model

# image test
@app.post("/img_test")
async def test_model_endpoint(files: list[UploadFile] = File(...),
                              model: bytes = File(...)):

    # 다른 post에서 생성한 전역변수
    global encoder
    global device
    print(type(model))
    # model = (BytesIO(model))
    model = torch.load(BytesIO(model))
    model.eval()

    # predict test_img 
    for file in files:
        image = Image.open(BytesIO(await file.read()))
        image = transform_img(image)
        img_tensor = image.unsqueeze(0)  # 배치 차원 추가
        img_tensor = img_tensor.to(device)

        output = model(img_tensor)
        pred_prob = torch.softmax(output, dim=1)[0]  # 소프트맥스 함수를 통해 예측 확률 계산
        pred_label = torch.argmax(pred_prob).item()  # 가장 높은 확률을 가진 클래스 라벨 추출
        pred_encoder_label = encoder.inverse_transform([torch.argmax(pred_prob).item()])[0]

    return {'pred_label':pred_encoder_label, 'prob':pred_prob[pred_label].item()}