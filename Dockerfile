FROM tensorflow/tensorflow:2.5.0-gpu-jupyter 

WORKDIR /app
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install vim

COPY . .