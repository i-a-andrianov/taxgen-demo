from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='15WsRd6Bh8TAuEnkCgadDvOm5eyuT8lxJ',
                                    dest_path='./src/projection_model_gb-b.pt',
                                    unzip=False)

gdd.download_file_from_google_drive(file_id='1PKzG2OsIsQICM2vare5k6_TplMhjs1eZ',
                                    dest_path='./src/node_graph_reconstruct_model_directed.txt',
                                    unzip=False)
