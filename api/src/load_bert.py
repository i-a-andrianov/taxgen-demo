from google_drive_downloader import GoogleDriveDownloader as gdd
import gdown

url = 'https://drive.google.com/uc?id=1hM3byvh3iDRWAzZ6mmAt3-C5U1gZfRmy'
output = './src/node_graph_reconstruct_model_directed.txt'
gdown.download(url, output, quiet=False)


gdd.download_file_from_google_drive(file_id='1OiZOaA2pKTmyJbDpQPds3QYeBhZw6fC-',
                                    dest_path='./src/projection_model_gb-b.pt',
                                    unzip=False)


