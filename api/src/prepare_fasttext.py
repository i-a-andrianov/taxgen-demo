import gzip
import shutil

import wget

if __name__ == '__main__':
    wget.download("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz")

    with gzip.open('cc.en.300.bin.gz', 'rb') as f_in:
        with open('cc.en.300.bin', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
