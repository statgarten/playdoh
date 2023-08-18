# :yellow_heart: playdoh <img src="logo.png" width="120" align="right"/>

<!-- badges: start -->
[![Lifecycle:experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->
## What is playdoh?
The playdoh package provides some applications that can be used easily by people who are not familiar with AI. This package offers the following features:

- Image Classification
- Sentiment Analysis
- Speech-to-Text
- Time Series Forecasting

## Download models
The pretrained models required for using the Playdoh package are almost 1 GB in size, but we are not using the paid plan of git lfs. **Please download the models from the link below and place them in the 'fastapi_backend/pretrained_model' folder.**

[Download models (Google Drive)](https://drive.google.com/drive/folders/1up4XtIwaaLf_bUAQxGlbqGaNU6Lq-I0T?usp=drive_link)

## Prerequisite (w/ Anaconda)
- [Anaconda](https://www.anaconda.com/download) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- Visual Studio Code

## Development Stack
- Python
- Streamlit
- FastAPI
- Pytorch

## Instruction for Dev
1. Launch Visual Studio Code (VS Code):
    - Navigate to the playdoh directory and open it in VS Code.
2. Set the Python Interpreter:
    - Use the shortcut `Ctrl + Shift + P` to open the command palette.
    - Search for and select `Python: Select Interpreter`.
    - From the list, choose the interpreter associated with Anaconda or Miniconda.
3. Configure conda-forge Channels:
    - Execute the following commands:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
```
4. Create Virtual Environments:
    - Set up two distinct virtual environments with Python version 3.9.17 for frontend and backend:
```
conda create -n playdoh_front python=3.9.17
conda create -n playdoh_back python=3.9.17
conda env list
```
5. Setup and Run Frontend & Backend:
    - For Frontend:
    ```
    conda activate playdoh_front
    cd ./streamlit_frontend
    pip install -r requirements_front.txt
    streamlit run app.py
    ```
    - For Backend:

    ```
    conda activate playdoh_back
    cd ./fastapi_backend
    pip install -r requirements.txt
    conda install libsndfile ffmpeg
    uvicorn server:app --reload --host=127.0.0.1 --port=8500
    ```
6. Once set up, navigate to `127.0.0.1:8501` in your browser.
