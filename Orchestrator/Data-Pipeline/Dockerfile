FROM python:3.10.15-slim-bookworm

WORKDIR /Data-Pipeline

ADD . /Data-Pipeline/data

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]