import hashlib
import time
from openslide import OpenSlide
from io import BytesIO
import os

def work():
    path = 'F:\\tif_images\\thyroid\\'
    for root, dirs, files in os.walk(path):
        print(files)



if __name__ == '__main__':
    work()
