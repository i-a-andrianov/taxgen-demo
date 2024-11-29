FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/prepare_nltk.py ./src/prepare_nltk.py
RUN python src/prepare_nltk.py

COPY src/load_images.py ./src/load_images.py
RUN python src/load_images.py

COPY src/load_bert.py ./src/load_bert.py
RUN python src/load_bert.py

RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*
RUN unzip ./src/wn_images.zip -d ./src/

COPY src ./src
ENV PYTHONPATH=src
CMD ["python", "src/main.py"]
