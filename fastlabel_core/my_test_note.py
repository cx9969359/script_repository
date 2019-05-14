import hashlib
import time
from openslide import OpenSlide
from io import BytesIO
import os
import numpy as np
import pyvips
import cv2
import xml.etree.ElementTree as ET


# map vips formats to np dtypes


def work():
    input_file_path = 'F:/working/split_image/label_xml'
    file_list = []
    for root, dirs, files in os.walk(input_file_path):
        for f in files:
            file_list.append(os.path.splitext(os.path.join(root, f).replace('\\', '/').replace(
                os.path.join(input_file_path, '').replace('\\', '/'), ''))[0])
    print(file_list)
    return file_list

if __name__ == '__main__':
    work()
