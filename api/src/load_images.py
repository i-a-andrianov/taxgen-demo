import gdown

url = 'https://drive.google.com/uc?id=1VNWluYhO5jOxu6qO_FhJKR7a2NkMFOdJ'
output = 'src/wn_images.zip'
gdown.download(url, output, quiet=False)
