<!-- badges: start -->
[![Lifecycle:experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->
## What is playdoh?
The playdoh package provides some applications that can be used easily by people who are not familiar with AI. This package offers the following features:

- Image Classification
- Sentiment Analysis From Text
- Speech-to-Text Conversion
- Time Series Forecasting

## Prerequisite
- Docker
- Visual Studio Code

## Development Stack
- Python
- Streamlit
- FastAPI
- Pytorch

## Instruction for Dev
1. Clone playdoh and open it in VS Code.
2. Install the 'Docker' and 'Dev Containers' extensions.
3. Press F1(or Ctrl(Cmd) + Shift + P) and select 'Dev-Containers: Add Dev Container Configuration Files' -> 'Show All Definitions' -> 'Python3', then choose your desired Python version and select any additional features to use.
4. If you wish to install additional python packages, add them to requirements.txt.
5. Press F1(or Ctrl(Cmd) + Shift + P) and select 'Dev-Containers: Rebuild Container'.
6. Open 2 terminals of the container and run the following commands:
```
## terminal 1
uvicorn server:app --reload
```
```
## terminal 2
streamlit run app.py
```
 
