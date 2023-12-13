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
2. Download pretrained models into `fastapi_backend/pretrained_model` from [Google Drive](https://drive.google.com/drive/folders/1up4XtIwaaLf_bUAQxGlbqGaNU6Lq-I0T?usp=drive_link)
3. Build Docker images and run containers using Docker Compose with the following command:
    ```
    docker-compose up -d
    ```
4. Once set up, navigate to `127.0.0.1:80` or `localhost:80` in your browser.
5. If you make any modifications to the code, either on the frontend or backend, restart the respective container derived from the corresponding image.

## Task tutorial
Click the image below to view other tutorial videos.

[![Video Label](http://img.youtube.com/vi/yUz0V6R-EqA/0.jpg)](http://youtu.be/yUz0V6R-EqA)

## Authors

-   Dongryul Min [\@MinDongRyul](http://github.com/MinDongRyul)

## License

Copyright :copyright: 2023 PHI This project is [MIT](https://opensource.org/licenses/MIT) licensed
