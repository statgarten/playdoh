FROM python:3.9-slim-buster

WORKDIR /app
ADD . /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001
EXPOSE 8501

CMD ["bash", "start.sh"]
