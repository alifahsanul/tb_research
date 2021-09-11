FROM python:3.6-slim-buster

COPY deploy/requirements.txt requirements.txt
RUN pip3 install -r /requirements.txt

COPY . .

CMD [ "python3", "deploy/app.py", "--host=0.0.0.0"]