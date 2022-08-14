FROM python:3.9
COPY . /bots_talk/
WORKDIR /bots_talk
RUN apt-get update
RUN pip install -r requirements.txt 
EXPOSE 5000 