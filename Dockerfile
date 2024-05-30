FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    python3-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean

RUN pip install --upgrade pip

RUN pip install tensorflow-cpu==2.16.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY happy_sad_model.keras /app/

EXPOSE 8088

COPY .env .env
RUN export $(cat .env | xargs)

CMD ["python", "main.py"]
