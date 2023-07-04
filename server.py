import torch
import torch.nn.init
import torch.nn as nn
import torch.cuda as tc
import torch.optim as optim # 최적화 함수
import torchvision.transforms as transforms # 전처리

from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse

from starlette.responses import FileResponse

from PIL import Image
from io import BytesIO

# MobilenetV3를 구성하기 위해 py와 가중치파일을 따로 폴더에 넣기
import pretrained_model.mobilenet_v3.mobilenetv3 as mobilenetv3

import numpy as np
import pandas as pd

#sa
from fastapi import FastAPI, Request
from transformers import AutoModelForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer
#stt
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import librosa

app = FastAPI()

######## Image Calssification ########
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
    weight_path = "pretrained_model/mobilenet_v3/mobilenetv3-large.pth"
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
@app.get("/img_classification_model_download")
def download_img_model():

    model_path = 'trained_model/image_classification_model.pth'

    return FileResponse(path=model_path, filename=model_path, media_type='application/octet-stream')

######## Image Calssification End ########

########### Time Forecasting #############
# 결측치 제거
def drop_null(df):
    if sum(df.isnull().sum()) > 0:
        df = df.dropna()
    return df

# train data과 test data 분리
def train_test_data_split(df, train_s, train_e, test_s, test_e, date, col, label_data):
    
    train = df[train_s : train_e]
    test = df[test_s : test_e]
    
    train = train[[date, col, col]]
    test = test[[date, col, col]]
    
    train.columns = [date, col, label_data]
    test.columns = [date, col, label_data]
    
    train = train.set_index(date)
    test = test.set_index(date)
    
    x_train = train[[col]]
    x_test = test[[col]]
    y_train = train[[label_data]]
    y_test = test[[label_data]]
    
    return x_train, y_train, x_test, y_test

# 정답 데이터와 입력 데이터 분리
# 윈도우 사이즈 별로 스케일링
# series_data, window_size, horizon, 예측컬럼, 날짜컬럼
# X, Y = seq2dataset(feature_np, w, h)
def seq2dataset(input_data, label_data, window, horizon):
    
    X = [] # 입력 데이터를 저장하는 list
    Y = [] # 정답 데이터를 저장하는 list
 
    # seq = feature_np / feature_np -> feature_np = scaled_df[int_col]
    for i in range(len(input_data)-(window+horizon) + 1):
        # window_size만 끊어서 정규화하여 입력 데이터로 분리
        # 여기서 x는 60개의 데이터를 뽑아냄
        # data_df[[col]][i:(i+window)] 슬라이싱 이용하여 [[..], [..], [..]] 형상으로 X데이터를 생성함
        x = input_data[i:(i+window)]
        y = label_data[i+window+horizon-1]
        X.append(x)
        Y.append(y)
    
    return np.array(X), np.array(Y)
    # x.shape = [[...], [...], [...], ...] 은 2차원 행렬인데, np.array(X)를 통해서 
    # (batch_size, time_steps, input_dims) 형상을 가지는 3차원 텐서로 변환되어 리턴

# Numpy array상태로는 학습이 불가능하므로, Torch Variable 형태로 변경(data/grad/grad_fn)
def numpy_to_torch(train_x, train_y, test_x, test_y):
    
    train_x_tensor = Variable(torch.Tensor(train_x)) # torch.Tensor
    train_y_tensor = Variable(torch.Tensor(train_y))

    test_x_tensor = Variable(torch.Tensor(test_x))
    test_y_tensor = Variable(torch.Tensor(test_y))

    return train_x_tensor, train_y_tensor, test_x_tensor, test_y_tensor

# 모델 설계
# LSTM_이라는 class가 nn.Module이라는 부모클래스를 상속받고 있는 상태
class LSTM_(nn.Module):
    # input_dim, hidden_dim, num_layers, output_dim, device 를 입력으로 받는 초기화 함수
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        # 파생클래스와 self를넣어서 현재 클래스가 어떤 클래스인지 명확하게 표시
        # super().__init__()과는 기능적으로 차이가 없다
        super(LSTM_, self).__init__() # super().__init__() == super(파생클래스이름, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out,(hn, cn)=self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        return h0, c0

# 모델 훈련
def train_time_model(learning_rate, num_epochs, train_x_tensor, train_y_tensor, device):
    
    global LSTM_
    
    model = LSTM_(input_dim=1,hidden_dim=128,output_dim=1,num_layers=2, device=device).to(device)

    loss_check = []
    
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    best_loss = 10 ** 9 # 최대한 높게 선정
    patience_limit = 30 # 몇 번의 epoch까지 지켜볼지를 결정
    patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
    
    for epoch in range(num_epochs): 

        # outputs = LSTM_.forward(train_x_tensor_final.to(device))
        outputs = model(train_x_tensor.to(device))

        optimizer.zero_grad()

        loss = loss_function(outputs, train_y_tensor.to(device))
        loss_check.append(loss.item())
        
        loss.backward()

        optimizer.step() # improve from loss = back propagation

        print("Epoch : %d, loss : %1.5f" % (epoch+1, loss.item()))
            
        ### early stopping 여부를 체크하는 부분 ###
        if loss > best_loss: # loss가 개선되지 않은 경우
            patience_check += 1
            if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
                break
        else: # loss가 개선된 경우
            best_loss = loss
            patience_check = 0
            
    return model, loss_check

# 모델 테스트
def test_time_model(model, test_x_tensor, scaler, device):

    model.eval() # model eval모드
    
    # torch.no_grad()로 감싸진(with) 부분에서는 
    # gradient 계산을 하지 않아 메모리/속도가 개선
    with torch.no_grad():
    # Estimated Value
        test_predict = model(test_x_tensor.to(device))  #   #Forward Pass

    predict_data = test_predict.data.detach().cpu().numpy()  # numpy conversion
    # cpu로 이동시켜서 numpy배열로 변환하는 이유
    # 작은 규모의 연산이거나 메모리 요구 사항이 적은 경우 CPU가 더 효율적일 수 있는 가능성 존재
    # 따라서, GPU를 사용하여 예측을 수행한 후, 예측값을 가져와 역정규화하는 경우에는 CPU로 데이터를 이동시키는 것이 자연스럽고
    # 역정규화는 단일 예측값에 대해 수행되는 간단한 계산이므로 CPU로 데이터를 이동시키는 것이 더 효율적일 수 있다.
    # 또한, GPU 메모리를 확보하기 위해서도 CPU로 데이터를 이동시키는 경우도 존재.
    
    inverse_predict_data = scaler.inverse_transform(
        predict_data
    )  # inverse normalization(Min/Max)

    inverse_pred_lst = [i[0] for i in inverse_predict_data]
    pred_lst = [i[0] for i in predict_data]
    
    return inverse_pred_lst, pred_lst

# 추가적인 데이터 예측
def additional_pred(model, input_sequense, num_features, scaler, device):
    
    model.eval() # model eval모드

    # 메모리 문제 가능성이 존재하기에 초기 값 설정
    plus_test_tensor = torch.zeros(input_sequense.shape)
    
    # torch tensor에서 예측 데이터를 붙히는 과정
    for _ in range(num_features):
        with torch.no_grad():
            plus_predict = model(input_sequense.to(device))
        # dimension을 맞춰 data를 concat실행 후 줄인 차원을 다시 증가시키는 과정
        # [1:]로 하는 이유 : 첫번째 이후 데이터에 예측 데이터를 붙히기 위함
        plus_tensor = torch.cat([input_sequense.squeeze(0).to(device), plus_predict[[0]]], dim=0)[1:].unsqueeze(0)
        plus_test_tensor = torch.cat([plus_test_tensor.to(device), plus_tensor.to(device)])
        input_sequense = plus_tensor

    print('plus_test_tensor shape :' ,plus_test_tensor[1:].shape)
    
    with torch.no_grad():
        test_plus_predict = model(plus_test_tensor[1:].to(device))
    additional_predict_data = test_plus_predict.data.detach().cpu().numpy() # torch -> numpy conversion
    additional_plus_data = scaler.inverse_transform(additional_predict_data) # inverse normalization(Min/Max)

    print('additional_plus_data shape :', additional_plus_data.shape)
    print('additional_plus_data type :', type(additional_plus_data))
    
    return additional_plus_data

# training 
@app.post("/time_train")
async def time_train_endpoint(data_arranges:list[str],
                              pred_col: str = Form(...),
                              date: str = Form(...),
                              window_size: int = Form(...),
                              horizon_factor: int = Form(...),
                              epoch: int = Form(...),
                              learning_rate: float = Form(...),
                              file : UploadFile = File(...)):
    global device
    device = 'cuda' if tc.is_available() else 'cpu'

    label_data = 'Label_Data'

    content = await file.read()
    train_df = pd.read_csv(BytesIO(content), encoding='utf-8')

    train_df[date] = pd.to_datetime(train_df[date])

    # null remove
    train_df = drop_null(train_df)

    # data split
    x_train, y_train, x_test, y_test = train_test_data_split(train_df, 
                                                            int(data_arranges[0]), 
                                                            int(data_arranges[1]), 
                                                            int(data_arranges[2]), 
                                                            int(data_arranges[3]), 
                                                            date, 
                                                            pred_col, 
                                                            label_data) # DataFrame
    print('Training shape :', x_train.shape, y_train.shape)
    print('Testing shape :', x_test.shape, y_test.shape) 

    # 정규화
    x_scaler = MinMaxScaler()
    x_scaled_train = x_scaler.fit_transform(x_train)
    x_scaled_test = x_scaler.transform(x_test)

    global y_scaler
    y_scaler = MinMaxScaler()
    y_scaled_train = y_scaler.fit_transform(y_train)
    y_scaled_test = y_scaler.transform(y_test)

    print('train scaled shape :', x_scaled_train.shape, y_scaled_train.shape)
    print('test scaled shape :', x_scaled_test.shape, y_scaled_test.shape)

    # 훈련데이터와 테스트데이터 를 각각 정규화 및 정답데이터 훈련데이터 분리
    train_x, train_y = seq2dataset(x_scaled_train, y_scaled_train, window_size, horizon_factor) # numpy_array
    test_x, test_y = seq2dataset(x_scaled_test, y_scaled_test, window_size, horizon_factor)

    # test_x가 0개면 return?
    if test_x.shape[0] == 0 or train_x.shape[0] == 0:
        response_content = {
            'scaled_size': min(x_scaled_test.shape[0], x_scaled_train.shape[0]),
            'data_success': False
        }
        return JSONResponse(content = response_content) # -> error 표시와, x_scaled_test의 갯수를 리턴하여 이 갯수보다 줄이라고 표시해주기?

    # Check Data pre-processing
    print("Training numpy array shape :", train_x.shape, train_y.shape)
    print("Testing numpy array shape :", test_x.shape, test_y.shape)

    # Numpy array상태로는 학습이 불가능하므로, Torch Variable 형태로 변경(data/grad/grad_fn)
    # lstm의 input 형태로 변경 - batch first = True 이므로 torch.tensor(batch, seq, feature)
    train_x_tensor, train_y_tensor, test_x_tensor, test_y_tensor = numpy_to_torch(train_x, train_y, test_x, test_y)
    print("Training data tensor shape :", train_x_tensor.shape, train_y_tensor.shape)
    print("Testing data tensor shape :", test_x_tensor.shape, test_y_tensor.shape)

    # 모델 훈련
    time_series_model, loss_check = train_time_model(learning_rate, epoch, train_x_tensor, train_y_tensor, device)

    # 모델 가중치 저장
    torch.save(time_series_model, 'trained_model/time_series_forecasting.pth')
    
    response_content = {
        # return 하기 위해 numpy array후 list화 진행
        'test_x_tensor': test_x_tensor.numpy().tolist(),
        'data_success': True
    }
    # print(test_x_tensor.numpy().tolist())
    return JSONResponse(content = response_content)

# prediction
@app.post("/time_pred")
def time_pred_endpoint(test_x_tensor:list[str],
                       num_features: int = Form(...),
                       window_size: int = Form(...)):
    global device
    global y_scaler
    # model load
    time_series_model = torch.load("trained_model/time_series_forecasting.pth", map_location=device)
    # test_x_tensor 구조를 (540, 1) -> (135, 4, 1)로 바꾸기 위함
    test_x_tensor = torch.tensor([eval(i) for i in test_x_tensor])
    test_x_tensor = test_x_tensor.reshape(test_x_tensor.shape[0]//window_size, window_size, 1)

    # 모델 테스트
    pred_lst, inverse_pred_lst = test_time_model(time_series_model, test_x_tensor, y_scaler, device)

    # 추가적인 데이터 예측
    input_sequense = test_x_tensor[[-1]]
    predict_additional_data = additional_pred(time_series_model, input_sequense, num_features, y_scaler, device)

    pred_list = [float(i) for i in pred_lst]
    predict_additional_list = [float(i[0]) for i in predict_additional_data]

    return {'data_success':True, 'pred_list':pred_list, 'predict_additional_list':predict_additional_list}

# time forecasting model 다운로드
@app.get("/time_series_model_download")
def download_time_series_model():

    model_path = 'trained_model/time_series_forecasting.pth'

    return FileResponse(path=model_path, filename=model_path, media_type='application/octet-stream')
########### Time Forecasting end #############

########### Sentiment Analysis #############

@app.post("/sentiment_analysis")
async def predict_sentiment_endpoint(request: Request):
    labels = {0: '기쁨', 1: '우울', 2: '분노', 3: '두려움', 4: '사랑', 5: '놀람', 6: '중립'}
    model_name = "./pretrained_model/kobert_ft"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 7)
    tokenizer = KoBERTTokenizer.from_pretrained(model_name, truncation_side="left") 
    data = await request.json()
    text = data.get('text', '')

    inputs = tokenizer(text, truncation=True, padding=True, max_length = 512, return_tensors="pt")
    output = model(**inputs)
    logits = output.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    prob_to_numpy = probabilities.detach().cpu().numpy()[0] 
    sent_prob = {labels[i]: float(round(prob_to_numpy[i]*100, 2)) for i in range(len(labels))}  # define dictionary to make sentiment_analysis.py can get 'sent_prob' as JSON
    sentiment_predicted = labels[prob_to_numpy.argmax()] # most probable sentiment

    return {"sent_prob":sent_prob, "sentiment_predicted":sentiment_predicted}

########### Sentiment Analysis End #############

########### Speech2Text #############
import tempfile
import os
from scipy.signal import resample
from pydub import AudioSegment

def convert_audio_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.replace(".mp3", ".wav").replace(".wma", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

@app.post("/speech_to_text")
async def transcribe_endpoint(file: UploadFile = File(...)):
    model_name = "./pretrained_model/wav2vec2"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Save temporary audio file
    extension = os.path.splitext(file.filename)[1]  # Get the file extension
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    audio_file = await file.read()
    temp_audio_file.write(audio_file)
    temp_audio_file.close()
    
    wav_path = convert_audio_to_wav(temp_audio_file.name)
    audio, rate = librosa.load(wav_path, sr=None)

    if len(audio.shape) > 1: 
        audio = audio[:,0] + audio[:,1]
    if rate != 16000:
        num_samples = int(audio.shape[0] * 16000 / rate)
        audio = resample(audio, num_samples)

    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values # tokenize
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return {"transcription": transcription}
########### Speech2Text End #############
