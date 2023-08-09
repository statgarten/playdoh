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

## Checking for Git LFS
This repository uses Git Large File Storage (LFS) to handle the tracking of pretrained model objects. If you do not have Git LFS installed, you can install it on this page([Git LFS page](https://git-lfs.com/)).

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
2. Build Docker images and run containers using Docker Compose with the following command:
```
docker-compose up -d
```
3. If you make any modifications to the code, either on the frontend or backend, restart the respective container derived from the corresponding image.
