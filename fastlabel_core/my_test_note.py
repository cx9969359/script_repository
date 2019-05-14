import hashlib
import time
from openslide import OpenSlide
from io import BytesIO
import os
import argparse
import numpy as np
import pyvips
import cv2
import xml.etree.ElementTree as ET


# map vips formats to np dtypes


def work():
    parser = argparse.ArgumentParser(description='Generate tile image from annotation xml',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_file_path = 'F:/create_seg_config.yml'
    args = parser.parse_args()
    yml_file = open(args.yml_path, encoding='utf-8')

if __name__ == '__main__':
    work()
