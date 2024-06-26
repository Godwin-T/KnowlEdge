FROM python:3.12-slim-bullseye

RUN apt-get update && apt-get install -y gcc zlib1g-dev gzip libjpeg-dev

RUN pip install -U pip

WORKDIR /home

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ["helperfunctions.py", "KnowlEdge.py", "HtmlTemplate.py", "./"]

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "KnowlEdge.py", "--server.port=8501", "--server.address=0.0.0.0"]
