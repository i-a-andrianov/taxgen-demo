FROM python:3.9.21

WORKDIR /app

COPY requirements.txt .
RUN pip install uv
RUN pip install -r requirements.txt

COPY src/prepare_nltk.py ./src/prepare_nltk.py
RUN python src/prepare_nltk.py

# COPY src/load_images.py ./src/load_images.py
# RUN python src/load_images.py
COPY src/wn_images.zip ./src/wn_images.zip
RUN unzip ./src/wn_images.zip -d ./src/

COPY src ./src
ENV PYTHONPATH=src
CMD ["python", "src/main.py"]
EXPOSE 60444
