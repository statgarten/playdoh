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
1. Open the playdoh folder in Visual Studio Code.
2. Press Ctrl + Shift + P and then choose Python: Select Interpreter. Select the interpreter that's based on Anaconda (or Miniconda).
3. Add the conda-forge channels by running:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
```
4. Install Python 3.9.17 and create two virtual environments for the frontend and backend respectively:
```
conda install python=3.9.17
conda create -n playdoh_front python=3.9.17
conda create -n playdoh_back python=3.9.17
conda env list
```
5. For each virtual environment, install the required dependencies:
```
# For the frontend
conda activate playdoh_front
pip install -r ./streamlit_frontend/requirements.txt

# For the backend
conda activate playdoh_back
pip install -r ./fastapi_backend/requirements.txt
```
6. Once set up, navigate to `127.0.0.1:8501` in your browser.
