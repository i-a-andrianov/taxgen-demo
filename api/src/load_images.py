import gdown

url = 'https://drive.google.com/uc?id=1T9iiNkp2X8n-W-N2MRZ6waQ0OkUulDzy'
output = 'src/wn_images.zip'
gdown.download(url, output, quiet=False)
