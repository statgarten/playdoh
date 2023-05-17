from fastapi import FastAPI, UploadFile, File
from torch import nn, optim

app = FastAPI()

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

@app.post("/img_train")
async def train_model_endpoint(images: dict[str, list[UploadFile]] = File(...)):
    # TODO: 
    # 1. Image transform
    # 2. Get image, label information
    # 3. Train the model with HP

    return {"message": "Model trained completed"}