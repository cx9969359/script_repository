import hashlib
import time
from openslide import OpenSlide
from io import BytesIO
import os
import numpy as np


def work():
    # path = 'F:\\tif_images\\thyroid\\more.tif'
    # slide = OpenSlide(path)
    # image_shape = (slide.dimensions[1],slide.dimensions[0])
    # print(image_shape)
    path = 'E:\\L201903670.tif.txt'
    a = {'hsil': [], 'lsil': [[1, 23, 3, 45], [3, 4, 665, 65]]}
    for category, bbox_list in a.items():
        print(category)
        for bbox in bbox_list:
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            coordinate_list = [{'x': x_min, 'y': y_min}, {'x': x_min, 'y': y_max}, {'x': x_max, 'y': y_min},
                               {'x': x_max, 'y': y_max}]



if __name__ == '__main__':
    work()
