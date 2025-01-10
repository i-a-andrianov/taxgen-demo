# HOW TO RUN BACKEND

1. Set up .env file in the root directory of the project:

```bash
touch api/src/.env
```

2. Add the following variables to the .env file:

```bash
CUDA_VISIBLE_DEVICES=1
# might be not needed for the latest model
HF_TOKEN=<token for accessing VityaVitalich/TaxoLlama3.1-8b-instruct>
```

3. Build and run docker image.

```bash
cd api
sudo docker build -t taxgen .
sudo docker run -gpus:all 
sudo docker run --gpus all -p 60444:8888 # 60444 is the open port of the server, 8888 is the port of the container  
```