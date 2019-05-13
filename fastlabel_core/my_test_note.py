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
    # path = 'F:/working/split_image/label_xml/V201803956LSIL_2019_01_28_15_26_39.xml'
    # annotation_xml_tree = ET.parse(path)
    # objects = annotation_xml_tree.findall('object')
    # for index, object in enumerate(objects):
    #     points = object.find('segmentation').findall('points')
    #     tuple_list = []
    #     for p in points:
    #         point_tuple = (p.find('x').text, p.find('y').text)
    #         tuple_list.append(point_tuple)
    #     array = np.asarray(tuple_list)
    #     print(array)
    a = {'a': 1}
    for k, v in a.items():
        print(k)
        print(v)


if __name__ == '__main__':
    work()
