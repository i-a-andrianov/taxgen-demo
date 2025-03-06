FROM python:3.9.21

WORKDIR /app

COPY requirements.txt .
RUN pip install uv
RUN pip install -r requirements.txt

COPY src/prepare_nltk.py ./src/prepare_nltk.py
RUN python src/prepare_nltk.py

RUN wget https://huggingface.co/datasets/lilaspourpre/ImageNet-Gen/resolve/main/data.zip
COPY src/ImageNet-Gen/data.zip ./src/ImageNet-Gen/data.zip
RUN unzip -j ./src/ImageNet-Gen/data.zip -d ./src/ImageNet-Gen/ && rm ./src/ImageNet-Gen/data.zip
RUN unzip ./src/wn_images.zip -d ./src/

COPY src ./src
ENV PYTHONPATH=src
CMD ["python", "src/main.py"]
EXPOSE 60444
