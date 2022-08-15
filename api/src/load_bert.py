import gdown

url = 'https://drive.google.com/u/1/uc?id=15WsRd6Bh8TAuEnkCgadDvOm5eyuT8lx&export=download'
output = 'projection_model_gb-b.pt'
gdown.download(url, output, quiet=False, fuzzy=True)


url = "https://drive.google.com/u/1/uc?id=1PKzG2OsIsQICM2vare5k6_TplMhjs1eZ&export=download"
output = 'node_graph_reconstruct_model_directed.txt'
gdown.download(url, output, quiet=False, fuzzy=True)


