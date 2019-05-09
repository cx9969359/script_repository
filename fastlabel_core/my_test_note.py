import hashlib
import time
from openslide import OpenSlide
from io import BytesIO
import os
import numpy as np
import pyvips
import cv2

# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}


def work():
    path = 'F:\\tif_images\\thyroid\\more.tif'
    pyvips_image = pyvips.Image.new_from_file(path)
    patch = pyvips_image.extract_area(900,900,1800,1800)
    return patch


def vips2numpy(vi):
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


if __name__ == '__main__':
    pyvips_image = work()
    np_arr = vips2numpy(pyvips_image)
    print(np_arr)
    print('============')
    img_mat = np.full((np_arr.shape[0], np_arr.shape[1], 1), float(0))
    print(img_mat)
