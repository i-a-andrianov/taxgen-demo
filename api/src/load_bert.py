import gdown

url = 'https://drive.google.com/uc?id=1hM3byvh3iDRWAzZ6mmAt3-C5U1gZfRmy'
output = './src/node_graph_reconstruct_model_directed.txt'
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/uc?id=1OiZOaA2pKTmyJbDpQPds3QYeBhZw6fC-'
output = './src/projection_model_gb-b.pt'
gdown.download(url, output, quiet=False)



