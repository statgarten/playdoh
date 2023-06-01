import torch.nn.init
import torch.nn as nn
import torch.cuda as tc
import torch.optim as optim # 최적화 함수
import torchvision.transforms as transforms # 전처리

from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder

from fastapi import FastAPI, UploadFile, File, Form 
from starlette.responses import FileResponse

from PIL import Image
from io import BytesIO

# MobilenetV3를 구성하기 위해 py와 가중치파일을 따로 폴더에 넣기
import mobilenet_v3.mobilenetv3 as mobilenetv3
import numpy as np

app = FastAPI()

# LabelEncdoing
def label_encoding(labels):
    
    encoder = LabelEncoder()
    encoder.fit(np.array(labels))
    train_labels = encoder.transform(np.array(labels))
    
    return encoder, train_labels

# preprcessing image
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[R채널 평균, G채널 평균, B채널 평균] , std=[R채널 표준편차, G채널 표준편차, B채널 표준편차])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ImageNet이 학습한 수백만장의 이미지의 RGB 각각의 채널에 대한 평균은 0.485, 0.456, 0.406 그리고 표준편차는 0.229, 0.224, 0.225
])

# create model
def create_model(num_classes, device):

    # 가중치 파일 경로
    weight_path = "mobilenet_v3/mobilenetv3-large.pth"
    # mobilenetv3_large 넣어주기
    model = mobilenetv3.mobilenetv3_large()
    # 가중치 파일 삽입?
    model.load_state_dict(torch.load(weight_path))

    for param in model.parameters():
        param.requires_grad = False
          
    # 마지막 레이어의 입력 채널 수를 가져옴
    last_channel = model.classifier[-1].in_features

    # 새로운 선형 레이어를 생성하여 마지막 레이어를 대체
    model.classifier[-1] = nn.Linear(last_channel, num_classes)
    
    return model.to(device)

# trian model
def train_model(learning_rate, num_epochs, dataloader, device, model, opti):

    criterion = nn.CrossEntropyLoss().to(device)
    if opti == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif opti == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif opti == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

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

    global device
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
    model = train_model(learning_rate, epoch, dataloader, device, model, opti)

    # 저장 시 trained model 폴더로
    torch.save(model, 'trained_model/image_classification_model.pth')

    return model

# image test
@app.post("/img_test")
async def test_model_endpoint(files: list[UploadFile] = File(...)):

    # 다른 post에서 생성한 전역변수
    # global model

    global encoder
    global device
    # model load
    model = torch.load("trained_model/image_classification_model.pth", map_location=device)
    model.eval()

    # predict test_img 
    for file in files:
        image = Image.open(BytesIO(await file.read()))
        image = transform_img(image)
        img_tensor = image.unsqueeze(0)  # 배치 차원 추가
        img_tensor = img_tensor.to(device)

        output = model(img_tensor)
        pred_prob = torch.softmax(output, dim=1)[0]  # 소프트맥스 함수를 통해 예측 확률 계산
    
    pred_list = [] # return할 예측 리스트
    for idx, ratio in enumerate(pred_prob):
        pred_list.append((encoder.inverse_transform([idx])[0], ratio.item())) # (label, ratio)

    print(pred_prob)
    print(pred_list)
    
    # 확률이 높은 순으로 정렬
    pred_list.sort(key = lambda x : x[1], reverse=True)

    return {'prediction':pred_list}


# model download endpoint
@app.get("/model_download")
def download_model():

    model_path = 'trained_model/image_classification_model.pth'

    return FileResponse(path=model_path, filename=model_path, media_type='application/octet-stream')

@app.post("/time_train")
async def time_train_endpoint(file : UploadFile = File(...)):
    
    print(BytesIO(await file.read()))
    return {'message':'success'}