from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1OiZOaA2pKTmyJbDpQPds3QYeBhZw6fC-',
                                    dest_path='./src/projection_model_gb-b.pt',
                                    unzip=False)

gdd.download_file_from_google_drive(file_id='1hM3byvh3iDRWAzZ6mmAt3-C5U1gZfRmy',
                                    dest_path='./src/node_graph_reconstruct_model_directed.txt',
                                    unzip=False)
